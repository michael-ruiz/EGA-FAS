#!/usr/bin/env python3
"""
Unified Jetson benchmark for LFAS models.

Runs two benchmark modes from one entrypoint:
1. Model-only FPS: synthetic input, no camera.
2. Full pipeline FPS: capture -> detect -> preprocess -> infer -> smooth.

Supports three inference backends:
- PyTorch checkpoint (.pth)
- ONNX Runtime session (.onnx)
- TensorRT engine (.trt / .engine)
"""

import argparse
import collections
import importlib.util
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from benchmark_fps import MinimalConfig, detect_adaptive_guidance, infer_model_from_checkpoint, load_checkpoint
from model.bulid_model import get_model


PROJECT_ROOT = Path(__file__).resolve().parent
REALTIME_PATH = PROJECT_ROOT / "realtime testing" / "realtime_test.py"
_CV2 = None
_REALTIME_COMPONENTS = None
_REALTIME_IMPORT_ERROR = None


def get_cv2():
    global _CV2
    if _CV2 is None:
        try:
            import cv2 as _cv2
        except Exception as exc:
            raise RuntimeError(
                "OpenCV import failed. The container's cv2 build is missing a runtime "
                "dependency, so pipeline benchmarking is unavailable. "
                "Model-only benchmarking can still run with --skip_pipeline."
            ) from exc
        _CV2 = _cv2
    return _CV2


def load_realtime_module():
    spec = importlib.util.spec_from_file_location("realtime_test", REALTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import realtime module from {REALTIME_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_realtime_components():
    global _REALTIME_COMPONENTS, _REALTIME_IMPORT_ERROR
    if _REALTIME_COMPONENTS is None:
        try:
            module = load_realtime_module()
            _REALTIME_COMPONENTS = {
                "FaceDetector": module.FaceDetector,
                "FramePreprocessor": module.FramePreprocessor,
                "PredictionSmoother": module.PredictionSmoother,
                "RealSenseCapture": module.RealSenseCapture,
            }
        except Exception as exc:
            _REALTIME_IMPORT_ERROR = exc
            raise RuntimeError(
                "Realtime pipeline components could not be imported. "
                "This usually means OpenCV or RealSense dependencies are missing in the container."
            ) from exc
    return _REALTIME_COMPONENTS


@torch.inference_mode()
def run_pytorch_inference(model, tensor, device):
    outputs = model(tensor.to(device))
    logit = outputs[0]
    probs = F.softmax(logit, dim=1)
    spoof_prob = probs[0, 1].item()

    guidance_weights = None
    if len(outputs) >= 6 and outputs[5] is not None:
        guidance_weights = outputs[5][0].detach().cpu().numpy()
    return spoof_prob, guidance_weights


def depth_validity(depth_crop_raw, depth_min, depth_max):
    valid = (depth_crop_raw >= depth_min) & (depth_crop_raw <= depth_max)
    return float(valid.sum()) / max(depth_crop_raw.size, 1)


class LocalTRTInferencer:
    def __init__(self, engine_path, fp16=False):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT/pycuda not available. Install pycuda, and use a Jetson image "
                "with TensorRT runtime libraries present."
            ) from exc

        self._cuda = cuda
        self._fp16 = fp16
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as handle:
            self._engine = runtime.deserialize_cuda_engine(handle.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        self._in_bufs = []
        self._out_bufs = {}
        self._bindings = []

        for index in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(index)
            shape = tuple(self._engine.get_binding_shape(index))
            np_dtype = trt.nptype(self._engine.get_binding_dtype(index))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, np_dtype)
            dev = cuda.mem_alloc(host.nbytes)
            self._bindings.append(int(dev))
            if self._engine.binding_is_input(index):
                self._in_bufs.append({"host": host, "device": dev, "shape": shape})
            else:
                self._out_bufs[name] = {"host": host, "device": dev, "shape": shape}

    def infer(self, input_tensor):
        cuda = self._cuda
        inp = input_tensor.detach().cpu().numpy()
        inp = inp.astype(np.float16 if self._fp16 else np.float32, copy=False)

        np.copyto(self._in_bufs[0]["host"], inp.ravel())
        cuda.memcpy_htod_async(self._in_bufs[0]["device"], self._in_bufs[0]["host"], self._stream)
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        for out in self._out_bufs.values():
            cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)
        self._stream.synchronize()

        logit = self._out_bufs["logit"]["host"].reshape(self._out_bufs["logit"]["shape"])
        spoof_prob = compute_spoof_prob_from_logits(logit)

        guidance = None
        if "guidance_weights" in self._out_bufs:
            guidance = self._out_bufs["guidance_weights"]["host"].reshape(
                self._out_bufs["guidance_weights"]["shape"]
            )[0].astype(np.float32)
        return spoof_prob, guidance


class LocalPredictionSmoother:
    def __init__(self, window_size=5):
        self._buf = collections.deque(maxlen=window_size)

    def update(self, spoof_prob):
        self._buf.append(spoof_prob)

    def get(self):
        return float(np.mean(self._buf)) if self._buf else 0.5


class LocalFramePreprocessor:
    IMAGE_SIZE = 64

    def __init__(self, args):
        self.is_multi = args.is_multi
        self.image_modality = args.image_modality
        self.depth_min = args.depth_min
        self.depth_max = args.depth_max

    def _pad_and_crop(self, img, x, y, w, h, scale):
        ih, iw = img.shape[:2]
        cx, cy = x + w // 2, y + h // 2
        half_w = int(w * scale / 2)
        half_h = int(h * scale / 2)
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(iw, cx + half_w)
        y2 = min(ih, cy + half_h)
        return img[y1:y2, x1:x2]

    def _resize_hwc(self, img, size, mode):
        if img.ndim == 2:
            tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(tensor, size=(size, size), mode=mode)
            out = resized.squeeze(0).squeeze(0).cpu().numpy()
            return out.astype(img.dtype)

        tensor = torch.from_numpy(np.transpose(img.astype(np.float32), (2, 0, 1))).unsqueeze(0)
        resized = F.interpolate(tensor, size=(size, size), mode=mode)
        out = resized.squeeze(0).cpu().numpy()
        return np.transpose(out, (1, 2, 0)).astype(img.dtype)

    def _normalize_depth(self, depth_uint16):
        d = depth_uint16.astype(np.float32)
        d = np.clip(d, self.depth_min, self.depth_max)
        denom = max(self.depth_max - self.depth_min, 1)
        d = (d - self.depth_min) / denom * 255.0
        d = d.astype(np.uint8)
        return np.stack([d, d, d], axis=-1)

    def _to_3ch_gray(self, gray_uint8):
        return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    def _to_tensor(self, hwc_uint8):
        chw = np.transpose(hwc_uint8, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.from_numpy(chw).unsqueeze(0)

    def preprocess(self, color, depth, ir, x, y, w, h, scale=1.3):
        try:
            if self.is_multi:
                return self._preprocess_multi(color, depth, ir, x, y, w, h, scale)
            return self._preprocess_single(color, depth, ir, x, y, w, h, scale)
        except Exception:
            return None

    def _preprocess_multi(self, color, depth, ir, x, y, w, h, scale):
        sz = self.IMAGE_SIZE
        c_crop = self._pad_and_crop(color, x, y, w, h, scale)
        d_crop = self._pad_and_crop(depth, x, y, w, h, scale)
        i_crop = self._pad_and_crop(ir, x, y, w, h, scale)
        if c_crop.size == 0 or d_crop.size == 0 or i_crop.size == 0:
            return None

        color_64 = self._resize_hwc(c_crop, sz, "bilinear")
        depth_64_raw = self._resize_hwc(d_crop.astype(np.float32), sz, "nearest").astype(np.uint16)
        depth_64 = self._normalize_depth(depth_64_raw)
        ir_64 = self._to_3ch_gray(self._resize_hwc(i_crop, sz, "bilinear"))
        thermal_64 = np.zeros((sz, sz, 3), dtype=np.uint8)
        stacked = np.concatenate([color_64, depth_64, ir_64, thermal_64], axis=2)
        return self._to_tensor(stacked)

    def _preprocess_single(self, color, depth, ir, x, y, w, h, scale):
        sz = self.IMAGE_SIZE
        if self.image_modality == "color":
            crop = self._pad_and_crop(color, x, y, w, h, scale)
            if crop.size == 0:
                return None
            img = self._resize_hwc(crop, sz, "bilinear")
        elif self.image_modality == "depth":
            crop = self._pad_and_crop(depth, x, y, w, h, scale)
            if crop.size == 0:
                return None
            resized = self._resize_hwc(crop.astype(np.float32), sz, "nearest").astype(np.uint16)
            img = self._normalize_depth(resized)
        else:
            crop = self._pad_and_crop(ir, x, y, w, h, scale)
            if crop.size == 0:
                return None
            img = self._to_3ch_gray(self._resize_hwc(crop, sz, "bilinear"))
        return self._to_tensor(img)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark LFAS model-only FPS and full pipeline FPS on Jetson"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to PyTorch checkpoint (.pth) for the PyTorch backend.",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default=None,
        help="Path to ONNX model for the ONNX Runtime backend.",
    )
    parser.add_argument(
        "--trt_engine",
        type=str,
        default=None,
        help="Path to TensorRT engine (.trt / .engine) for the TensorRT backend.",
    )
    parser.add_argument(
        "--trt_fp16",
        action="store_true",
        default=False,
        help="TensorRT engine expects FP16 input.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ShffleNetV2_hd_v1_hybrid_d",
        help="PyTorch model architecture name. Auto-inferred from checkpoint path when possible.",
    )
    parser.add_argument(
        "--is_multi",
        action="store_true",
        default=False,
        help="Use multi-modal input (Color+Depth+IR+Thermal placeholder).",
    )
    parser.add_argument(
        "--image_modality",
        type=str,
        default="color",
        choices=["color", "depth", "ir"],
        help="Single-modal pipeline/input choice when --is_multi is not set.",
    )
    parser.add_argument(
        "--guidance_modality",
        type=str,
        default="depth",
        choices=["depth", "color", "ir"],
        help="Fixed guidance modality for PyTorch model construction.",
    )
    parser.add_argument(
        "--adaptive_guidance",
        action="store_true",
        default=False,
        help="Enable adaptive guidance for PyTorch model construction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device for PyTorch backend.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=64,
        help="Synthetic input image size for model-only benchmarking.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for model-only benchmarking.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=50,
        help="Warmup iterations for model-only benchmark.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Timed iterations for model-only benchmark.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="auto",
        choices=["auto", "realsense", "synthetic"],
        help="Input source for pipeline benchmark.",
    )
    parser.add_argument(
        "--pipeline_seconds",
        type=float,
        default=15.0,
        help="Timed duration in seconds for each pipeline benchmark.",
    )
    parser.add_argument(
        "--pipeline_warmup_frames",
        type=int,
        default=30,
        help="Warmup frames before timed pipeline measurement.",
    )
    parser.add_argument(
        "--detect_every",
        type=int,
        default=4,
        help="Run face detection every N frames and reuse the last box in between.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 for the PyTorch backend when running on CUDA.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Use torch.compile() for the PyTorch backend if available.",
    )
    parser.add_argument(
        "--skip_model_only",
        action="store_true",
        default=False,
        help="Skip model-only benchmarking.",
    )
    parser.add_argument(
        "--skip_pipeline",
        action="store_true",
        default=False,
        help="Skip full pipeline benchmarking.",
    )
    parser.add_argument(
        "--depth_min",
        type=int,
        default=300,
        help="Minimum depth in mm for normalization and depth-validity checks.",
    )
    parser.add_argument(
        "--depth_max",
        type=int,
        default=800,
        help="Maximum depth in mm for normalization and depth-validity checks.",
    )
    parser.add_argument(
        "--smooth_window",
        type=int,
        default=5,
        help="Rolling window size for prediction smoothing.",
    )
    parser.add_argument(
        "--face_scale",
        type=float,
        default=1.3,
        help="Scale factor for face crop padding.",
    )
    parser.add_argument(
        "--haar_scale",
        type=float,
        default=1.1,
        help="Haar cascade scaleFactor.",
    )
    parser.add_argument(
        "--haar_min_neighbors",
        type=int,
        default=5,
        help="Haar cascade minNeighbors.",
    )
    parser.add_argument(
        "--haar_min_size",
        type=int,
        default=60,
        help="Minimum face size for Haar detection.",
    )
    parser.add_argument(
        "--depth_validity_override",
        type=float,
        default=0.0,
        help="Spoof-probability boost threshold for low-validity depth crops.",
    )
    parser.add_argument(
        "--synthetic_width",
        type=int,
        default=424,
        help="Synthetic frame width for pipeline benchmarking.",
    )
    parser.add_argument(
        "--synthetic_height",
        type=int,
        default=240,
        help="Synthetic frame height for pipeline benchmarking.",
    )
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def get_input_channels(args):
    return 12 if args.is_multi else 3


def current_rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / 1024.0


def snapshot_tegrastats():
    tegrastats_path = shutil.which("tegrastats")
    if not tegrastats_path:
        return None

    proc = None
    try:
        proc = subprocess.Popen(
            [tegrastats_path, "--interval", "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        line = proc.stdout.readline().strip() if proc.stdout else ""
        return line or None
    except Exception:
        return None
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()


class MemoryTracker:
    def __init__(self, device):
        self.device = device
        self.peak_rss_mb = 0.0

    def reset(self):
        self.peak_rss_mb = current_rss_mb()
        if self.device.type == "cuda":
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def sample(self):
        self.peak_rss_mb = max(self.peak_rss_mb, current_rss_mb())

    def finalize(self):
        result = {"peak_rss_mb": self.peak_rss_mb}
        if self.device.type == "cuda":
            try:
                result["peak_gpu_allocated_mb"] = (
                    torch.cuda.max_memory_allocated(self.device) / (1024 * 1024)
                )
                result["peak_gpu_reserved_mb"] = (
                    torch.cuda.max_memory_reserved(self.device) / (1024 * 1024)
                )
            except Exception:
                result["peak_gpu_allocated_mb"] = None
                result["peak_gpu_reserved_mb"] = None
        else:
            result["peak_gpu_allocated_mb"] = None
            result["peak_gpu_reserved_mb"] = None
        result["tegrastats"] = snapshot_tegrastats()
        return result


def compute_spoof_prob_from_logits(logit):
    logits = np.asarray(logit, dtype=np.float32)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_l = np.exp(logits)
    probs = exp_l / exp_l.sum(axis=1, keepdims=True)
    return float(probs[0, 1])


class PyTorchBackend:
    def __init__(self, args, device):
        if not args.checkpoint:
            raise ValueError("PyTorch backend requires --checkpoint")

        self.kind = "pytorch"
        self.device = device
        self.fp16 = bool(args.fp16 and device.type == "cuda")
        self.model_name = args.model

        inferred_model = infer_model_from_checkpoint(args.checkpoint)
        if inferred_model and self.model_name == "ShffleNetV2_hd_v1_hybrid_d":
            self.model_name = inferred_model

        adaptive = args.adaptive_guidance
        adaptive = adaptive or detect_adaptive_guidance(args.checkpoint, device)

        config = MinimalConfig(
            model_name=self.model_name,
            is_multi=args.is_multi,
            guidance_modality=args.guidance_modality,
            adaptive_guidance=adaptive,
        )

        self.model = get_model(config, num_class=2)
        self.model = load_checkpoint(self.model, args.checkpoint, device)
        self.model = self.model.to(device).eval()

        if self.fp16:
            self.model = self.model.half()

        if args.compile:
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception as exc:
                print(f"[warn] torch.compile() failed for PyTorch backend: {exc}")

        precision = "fp16" if self.fp16 else "fp32"
        self.label = f"pytorch:{precision}"
        self.memory_device = device

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def make_random_input(self, input_shape):
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(input_shape, device=self.device, dtype=dtype)

    def infer(self, tensor):
        if self.fp16 and tensor.dtype != torch.float16:
            tensor = tensor.half()
        return run_pytorch_inference(self.model, tensor, self.device)

    def cleanup(self):
        del self.model
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


class ONNXBackend:
    def __init__(self, args, device):
        if not args.onnx:
            raise ValueError("ONNX backend requires --onnx")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "onnxruntime is not installed. Install onnxruntime-gpu on Jetson "
                "if you want CUDA-backed ONNX benchmarking."
            ) from exc

        self.kind = "onnx"
        self.device = device
        self.ort = ort

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if device.type == "cuda" and "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self.session = ort.InferenceSession(args.onnx, providers=providers)
        self.provider = self.session.get_providers()[0]
        self.input_name = self.session.get_inputs()[0].name
        self.input_type = self.session.get_inputs()[0].type
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.memory_device = (
            torch.device("cuda")
            if ("CUDAExecutionProvider" in self.provider and torch.cuda.is_available())
            else torch.device("cpu")
        )

        self.input_dtype = np.float16 if "float16" in self.input_type else np.float32
        dtype_str = "fp16" if self.input_dtype == np.float16 else "fp32"
        provider_str = "cuda" if "CUDAExecutionProvider" in self.provider else "cpu"
        self.label = f"onnx:{provider_str}-{dtype_str}"

    def sync(self):
        return None

    def make_random_input(self, input_shape):
        arr = np.random.randn(*input_shape).astype(self.input_dtype)
        return torch.from_numpy(arr)

    def infer(self, tensor):
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().cpu().numpy()
        else:
            arr = np.asarray(tensor)
        arr = arr.astype(self.input_dtype, copy=False)

        outputs = self.session.run(self.output_names, {self.input_name: arr})
        output_map = dict(zip(self.output_names, outputs))
        logit = output_map.get("logit", outputs[0])
        spoof_prob = compute_spoof_prob_from_logits(logit)

        guidance = None
        if "guidance_weights" in output_map:
            guidance = np.asarray(output_map["guidance_weights"], dtype=np.float32)[0]
        elif len(outputs) >= 6 and outputs[5] is not None:
            guidance = np.asarray(outputs[5], dtype=np.float32)[0]
        return spoof_prob, guidance

    def cleanup(self):
        del self.session


class TensorRTBackend:
    def __init__(self, args, device):
        if not args.trt_engine:
            raise ValueError("TensorRT backend requires --trt_engine")
        self.kind = "tensorrt"
        self.device = device
        self.fp16 = args.trt_fp16
        self.engine = LocalTRTInferencer(args.trt_engine, fp16=self.fp16)
        precision = "fp16" if self.fp16 else "fp32"
        self.label = f"tensorrt:{precision}"
        self.memory_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def sync(self):
        return None

    def make_random_input(self, input_shape):
        dtype = torch.float16 if self.fp16 else torch.float32
        return torch.randn(input_shape, dtype=dtype)

    def infer(self, tensor):
        return self.engine.infer(tensor)

    def cleanup(self):
        del self.engine


class SyntheticCapture:
    FPS = 30

    def __init__(self, width=424, height=240, depth_mm=550):
        self.W = width
        self.H = height
        self.depth_mm = depth_mm
        self.frame_index = 0
        self.default_face = (
            max(0, width // 2 - 60),
            max(0, height // 2 - 70),
            120,
            140,
        )

    def __enter__(self):
        print(f"Synthetic source started - {self.W}x{self.H}")
        return self

    def __exit__(self, *_):
        print("Synthetic source stopped.")

    def get_frames(self):
        color = np.full((self.H, self.W, 3), 25, dtype=np.uint8)
        depth = np.zeros((self.H, self.W), dtype=np.uint16)
        ir = np.full((self.H, self.W), 35, dtype=np.uint8)

        x, y, w, h = self.default_face
        cx = x + w // 2 + int(10 * np.sin(self.frame_index / 12.0))
        cy = y + h // 2 + int(6 * np.cos(self.frame_index / 14.0))

        face_x1 = max(0, cx - w // 2)
        face_y1 = max(0, cy - h // 2)
        face_x2 = min(self.W, cx + w // 2)
        face_y2 = min(self.H, cy + h // 2)
        color[face_y1:face_y2, face_x1:face_x2] = (190, 180, 160)

        eye_r = 8
        left_eye_y = max(0, cy - 18 - eye_r)
        left_eye_x = max(0, cx - 18 - eye_r)
        right_eye_y = max(0, cy - 18 - eye_r)
        right_eye_x = max(0, cx + 18 - eye_r)
        color[left_eye_y:left_eye_y + 2 * eye_r, left_eye_x:left_eye_x + 2 * eye_r] = (20, 20, 20)
        color[right_eye_y:right_eye_y + 2 * eye_r, right_eye_x:right_eye_x + 2 * eye_r] = (20, 20, 20)

        mouth_y1 = min(self.H, cy + 18)
        mouth_y2 = min(self.H, cy + 22)
        mouth_x1 = max(0, cx - 22)
        mouth_x2 = min(self.W, cx + 22)
        color[mouth_y1:mouth_y2, mouth_x1:mouth_x2] = (30, 30, 30)

        color[y:y + 2, x:x + w] = (60, 90, 140)
        color[y + h - 2:y + h, x:x + w] = (60, 90, 140)
        color[y:y + h, x:x + 2] = (60, 90, 140)
        color[y:y + h, x + w - 2:x + w] = (60, 90, 140)

        depth[y:y + h, x:x + w] = self.depth_mm
        depth[depth == 0] = 900
        ir[y:y + h, x:x + w] = 170

        self.frame_index += 1
        return color, depth, ir


def get_source_candidates(args):
    if args.source == "synthetic":
        return ["synthetic"]
    if args.source == "realsense":
        return ["realsense"]
    return ["realsense", "synthetic"]


def make_source(source_name, args):
    if source_name == "synthetic":
        return SyntheticCapture(args.synthetic_width, args.synthetic_height)
    components = get_realtime_components()
    return components["RealSenseCapture"]()


def build_pipeline_stack(source_name, args):
    if source_name == "synthetic":
        try:
            cv2 = get_cv2()
            components = get_realtime_components()
            return {
                "mode": "native",
                "cv2": cv2,
                "detector": components["FaceDetector"](
                    args.haar_scale, args.haar_min_neighbors, args.haar_min_size
                ),
                "preprocessor": components["FramePreprocessor"](args),
                "smoother": components["PredictionSmoother"](args.smooth_window),
                "source_name": "synthetic",
            }
        except Exception as exc:
            print(f"[info] OpenCV unavailable, using synthetic fallback pipeline: {exc}")
            return {
                "mode": "fallback",
                "cv2": None,
                "detector": None,
                "preprocessor": LocalFramePreprocessor(args),
                "smoother": LocalPredictionSmoother(args.smooth_window),
                "source_name": "synthetic-fallback",
            }

    cv2 = get_cv2()
    components = get_realtime_components()
    return {
        "mode": "native",
        "cv2": cv2,
        "detector": components["FaceDetector"](
            args.haar_scale, args.haar_min_neighbors, args.haar_min_size
        ),
        "preprocessor": components["FramePreprocessor"](args),
        "smoother": components["PredictionSmoother"](args.smooth_window),
        "source_name": source_name,
    }


def build_backends(args, device):
    backends = []
    errors = []

    if args.checkpoint:
        try:
            backends.append(PyTorchBackend(args, device))
        except Exception as exc:
            errors.append(f"PyTorch backend failed: {exc}")

    if args.onnx:
        try:
            backends.append(ONNXBackend(args, device))
        except Exception as exc:
            errors.append(f"ONNX backend failed: {exc}")

    if args.trt_engine:
        try:
            backends.append(TensorRTBackend(args, device))
        except Exception as exc:
            errors.append(f"TensorRT backend failed: {exc}")

    return backends, errors


def benchmark_backend(backend, input_data, tracker, warmup, iterations):
    latencies_ms = []
    tracker.reset()

    for _ in range(warmup):
        backend.sync()
        backend.infer(input_data)
        backend.sync()

    for _ in range(iterations):
        backend.sync()
        start = time.perf_counter()
        backend.infer(input_data)
        backend.sync()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
        tracker.sample()

    latencies = np.asarray(latencies_ms, dtype=np.float64)
    fps_values = 1000.0 / np.clip(latencies, a_min=1e-9, a_max=None)
    stats = {
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "mean_fps": float(np.mean(fps_values)),
    }
    stats.update(tracker.finalize())
    return stats


def center_face_bbox(frame_shape):
    height, width = frame_shape[:2]
    bw, bh = 120, 140
    x = max(0, width // 2 - bw // 2)
    y = max(0, height // 2 - bh // 2)
    return x, y, bw, bh


def run_pipeline_benchmark(args, backend, device):
    stage_names = ["capture", "detect", "preprocess", "infer", "smooth", "total"]
    tracker = MemoryTracker(backend.memory_device)

    timings = None
    frames_total = 0
    frames_with_face = 0
    frames_with_inference = 0
    source_name = None
    last_error = None

    for source_name_candidate in get_source_candidates(args):
        try:
            pipeline = build_pipeline_stack(source_name_candidate, args)
            cv2 = pipeline["cv2"]
            detector = pipeline["detector"]
            preprocessor = pipeline["preprocessor"]
            smoother = pipeline["smoother"]
            pipeline_mode = pipeline["mode"]
            timings = {name: [] for name in stage_names}
            frames_total = 0
            frames_with_face = 0
            frames_with_inference = 0
            last_faces = []
            source_name = pipeline["source_name"]
            source_obj = make_source(source_name_candidate, args)

            with source_obj as source:
                for _ in range(max(args.pipeline_warmup_frames, 0)):
                    try:
                        source.get_frames()
                    except Exception:
                        break

                tracker.reset()
                start_bench = time.perf_counter()

                while (time.perf_counter() - start_bench) < args.pipeline_seconds:
                    t_total = time.perf_counter()

                    t0 = time.perf_counter()
                    color, depth, ir = source.get_frames()
                    timings["capture"].append((time.perf_counter() - t0) * 1000.0)

                    t0 = time.perf_counter()
                    if frames_total % max(args.detect_every, 1) == 0:
                        if pipeline_mode == "native":
                            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                            last_faces = detector.detect(gray)
                        else:
                            last_faces = [center_face_bbox(color.shape)]
                    faces = last_faces

                    if not faces and source_name.startswith("synthetic"):
                        faces = [center_face_bbox(color.shape)]
                        last_faces = faces

                    timings["detect"].append((time.perf_counter() - t0) * 1000.0)

                    preprocess_ms = 0.0
                    infer_ms = 0.0
                    smooth_ms = 0.0

                    if faces:
                        frames_with_face += 1
                        x, y, w, h = faces[0]

                        t0 = time.perf_counter()
                        tensor = preprocessor.preprocess(
                            color, depth, ir, x, y, w, h, scale=args.face_scale
                        )
                        preprocess_ms = (time.perf_counter() - t0) * 1000.0

                        if tensor is not None:
                            t0 = time.perf_counter()
                            prob, _ = backend.infer(tensor)
                            infer_ms = (time.perf_counter() - t0) * 1000.0

                            d_crop = preprocessor._pad_and_crop(depth, x, y, w, h, args.face_scale)
                            val = depth_validity(d_crop, args.depth_min, args.depth_max)
                            thresh = args.depth_validity_override
                            if thresh > 0 and val < thresh:
                                blend = 1.0 - (val / thresh)
                                prob = prob + blend * (0.9 - prob)

                            t0 = time.perf_counter()
                            smoother.update(prob)
                            _ = smoother.get()
                            smooth_ms = (time.perf_counter() - t0) * 1000.0
                            frames_with_inference += 1

                    timings["preprocess"].append(preprocess_ms)
                    timings["infer"].append(infer_ms)
                    timings["smooth"].append(smooth_ms)
                    timings["total"].append((time.perf_counter() - t_total) * 1000.0)

                    frames_total += 1
                    tracker.sample()
            break
        except Exception as exc:
            last_error = exc
            if args.source == "auto" and source_name_candidate == "realsense":
                print(f"[info] RealSense unavailable, falling back to synthetic source: {exc}")
                continue
            raise

    if timings is None:
        raise RuntimeError(f"Pipeline benchmark could not start: {last_error}")

    total_elapsed = sum(timings["total"]) / 1000.0
    mean_total_ms = float(np.mean(timings["total"])) if timings["total"] else 0.0
    stats = {
        "source": source_name,
        "frames_total": frames_total,
        "frames_with_face": frames_with_face,
        "frames_with_inference": frames_with_inference,
        "elapsed_s": total_elapsed,
        "mean_fps": (frames_total / total_elapsed) if total_elapsed > 0 else 0.0,
        "mean_latency_ms": mean_total_ms,
        "p50_latency_ms": float(np.percentile(timings["total"], 50)) if timings["total"] else 0.0,
        "p95_latency_ms": float(np.percentile(timings["total"], 95)) if timings["total"] else 0.0,
        "stage_mean_ms": {name: float(np.mean(vals)) if vals else 0.0 for name, vals in timings.items()},
        "stage_pct": {
            name: (float(np.mean(vals)) / mean_total_ms * 100.0) if mean_total_ms > 0 and vals else 0.0
            for name, vals in timings.items()
        },
    }
    stats.update(tracker.finalize())
    return stats


def print_header(args, device, backends):
    channels = get_input_channels(args)
    print("=" * 72)
    print("LFAS Jetson Benchmark")
    print("=" * 72)
    print(f"Device:          {device}")
    print(f"Backends:        {', '.join(backend.label for backend in backends)}")
    print(f"Input channels:  {channels}")
    print(f"Model-only:      {'off' if args.skip_model_only else 'on'}")
    print(f"Pipeline:        {'off' if args.skip_pipeline else 'on'}")
    print(f"Pipeline source: {args.source}")
    if args.checkpoint:
        print(f"Checkpoint:      {args.checkpoint}")
    if args.onnx:
        print(f"ONNX:            {args.onnx}")
    if args.trt_engine:
        print(f"TensorRT:        {args.trt_engine}")
    print("=" * 72)


def print_model_only_result(label, stats):
    print(f"\n[model-only] {label}")
    print(f"  Mean FPS:       {stats['mean_fps']:.2f}")
    print(f"  Mean latency:   {stats['mean_latency_ms']:.2f} ms")
    print(f"  P50 latency:    {stats['p50_latency_ms']:.2f} ms")
    print(f"  P95 latency:    {stats['p95_latency_ms']:.2f} ms")
    print(f"  Peak RSS:       {stats['peak_rss_mb']:.1f} MB")
    if stats["peak_gpu_allocated_mb"] is not None:
        print(f"  Peak GPU alloc: {stats['peak_gpu_allocated_mb']:.1f} MB")
    if stats["peak_gpu_reserved_mb"] is not None:
        print(f"  Peak GPU resv:  {stats['peak_gpu_reserved_mb']:.1f} MB")
    if stats["tegrastats"]:
        print(f"  Tegrastats:     {stats['tegrastats']}")


def print_pipeline_result(label, stats):
    print(f"\n[pipeline] {label} ({stats['source']})")
    print(f"  Mean FPS:            {stats['mean_fps']:.2f}")
    print(f"  Mean frame latency:  {stats['mean_latency_ms']:.2f} ms")
    print(f"  P50 frame latency:   {stats['p50_latency_ms']:.2f} ms")
    print(f"  P95 frame latency:   {stats['p95_latency_ms']:.2f} ms")
    print(
        f"  Frames:              {stats['frames_total']} total, "
        f"{stats['frames_with_face']} with face, "
        f"{stats['frames_with_inference']} with inference"
    )
    for stage in ["capture", "detect", "preprocess", "infer", "smooth"]:
        print(
            f"  {stage:<20}"
            f"{stats['stage_mean_ms'][stage]:>7.2f} ms  "
            f"{stats['stage_pct'][stage]:>6.1f}%"
        )
    print(f"  Peak RSS:            {stats['peak_rss_mb']:.1f} MB")
    if stats["peak_gpu_allocated_mb"] is not None:
        print(f"  Peak GPU alloc:      {stats['peak_gpu_allocated_mb']:.1f} MB")
    if stats["peak_gpu_reserved_mb"] is not None:
        print(f"  Peak GPU resv:       {stats['peak_gpu_reserved_mb']:.1f} MB")
    if stats["tegrastats"]:
        print(f"  Tegrastats:          {stats['tegrastats']}")


def print_summary(results):
    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print(f"{'Backend':<20} {'Phase':<12} {'FPS':>10} {'P50 ms':>10} {'RSS MB':>10}")
    print("-" * 72)
    for row in results:
        print(
            f"{row['backend']:<20} {row['phase']:<12} "
            f"{row['fps']:>10.2f} {row['p50_ms']:>10.2f} {row['rss_mb']:>10.1f}"
        )


def main():
    args = parse_args()

    if args.skip_model_only and args.skip_pipeline:
        print("ERROR: both benchmark phases are disabled.")
        sys.exit(1)

    if not any([args.checkpoint, args.onnx, args.trt_engine]):
        print("ERROR: provide at least one of --checkpoint, --onnx, or --trt_engine.")
        sys.exit(1)

    device = resolve_device(args.device)
    backends, errors = build_backends(args, device)

    for error in errors:
        print(f"[warn] {error}")

    if not backends:
        print("ERROR: no backends could be initialized.")
        sys.exit(1)

    print_header(args, device, backends)

    summary_rows = []
    input_shape = (args.batch_size, get_input_channels(args), args.image_size, args.image_size)

    for backend in backends:
        try:
            if not args.skip_model_only:
                try:
                    tracker = MemoryTracker(backend.memory_device)
                    input_data = backend.make_random_input(input_shape)
                    stats = benchmark_backend(
                        backend, input_data, tracker, warmup=args.warmup, iterations=args.iterations
                    )
                    print_model_only_result(backend.label, stats)
                    summary_rows.append(
                        {
                            "backend": backend.label,
                            "phase": "model-only",
                            "fps": stats["mean_fps"],
                            "p50_ms": stats["p50_latency_ms"],
                            "rss_mb": stats["peak_rss_mb"],
                        }
                    )
                except Exception as exc:
                    print(f"\n[warn] model-only benchmark failed for {backend.label}: {exc}")

            if not args.skip_pipeline:
                try:
                    pipeline_stats = run_pipeline_benchmark(args, backend, device)
                    print_pipeline_result(backend.label, pipeline_stats)
                    summary_rows.append(
                        {
                            "backend": backend.label,
                            "phase": "pipeline",
                            "fps": pipeline_stats["mean_fps"],
                            "p50_ms": pipeline_stats["p50_latency_ms"],
                            "rss_mb": pipeline_stats["peak_rss_mb"],
                        }
                    )
                except Exception as exc:
                    print(f"\n[warn] pipeline benchmark failed for {backend.label}: {exc}")
        finally:
            backend.cleanup()

    print_summary(summary_rows)


if __name__ == "__main__":
    main()
