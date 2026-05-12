#!/usr/bin/env python3
"""
Real-time Face Anti-Spoofing with Intel RealSense Camera

Loads a trained LFAS model and runs live inference on RealSense D435/D435i
frames, displaying REAL/SPOOF predictions with bounding boxes and a confidence bar.

Usage:
    cd "/home/michael/LFAS-NewBackbone Ablation Study"

    # Multi-modal (best model, with depth visualization)
    python "realtime testing/realtime_test.py" \\
        --checkpoint results/WMCA/fusion/ShffleNetV2_hd_v1_hybrid_d_Multi_64/prot5/checkpoint/test_min_acer_model_XXXX.pth \\
        --is_multi --show_depth

    # Single-modal (color only)
    python "realtime testing/realtime_test.py" \\
        --checkpoint path/to/checkpoint.pth \\
        --image_modality color
"""

import argparse
import collections
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time Face Anti-Spoofing with Intel RealSense'
    )
    parser.add_argument(
        '--checkpoint', type=str, required=False, default=None,
        help='Path to .pth checkpoint file (required unless --trt_engine is used)'
    )
    parser.add_argument(
        '--model', type=str, default='ShffleNetV2_hd_v1_hybrid_d',
        help='Model architecture name — auto-inferred from checkpoint path when possible '
             '(default: ShffleNetV2_hd_v1_hybrid_d)'
    )
    parser.add_argument(
        '--is_multi', action='store_true', default=False,
        help='Use multi-modal model (Color+Depth+IR+Thermal). '
             'Default: single-modal'
    )
    parser.add_argument(
        '--image_modality', type=str, default='color',
        choices=['color', 'depth', 'ir'],
        help='Single-modal only: which modality to use (default: color)'
    )
    parser.add_argument(
        '--adaptive_guidance', action='store_true', default=False,
        help='Enable adaptive per-sample guidance — auto-detected from checkpoint'
    )
    parser.add_argument(
        '--guidance_modality', type=str, default='depth',
        choices=['depth', 'color', 'ir'],
        help='Fixed guidance modality when adaptive_guidance=False (default: depth)'
    )
    parser.add_argument(
        '--device', type=str, default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Compute device (default: auto)'
    )
    parser.add_argument(
        '--depth_min', type=int, default=300,
        help='Minimum depth in mm for normalization clipping (default: 300)'
    )
    parser.add_argument(
        '--depth_max', type=int, default=800,
        help='Maximum depth in mm for normalization clipping (default: 800)'
    )
    parser.add_argument(
        '--show_depth', action='store_true', default=False,
        help='Show a secondary window with colorized depth map'
    )
    parser.add_argument(
        '--smooth_window', type=int, default=5,
        help='Frames for rolling prediction average to reduce flicker (default: 5)'
    )
    parser.add_argument(
        '--face_scale', type=float, default=1.3,
        help='Scale factor for face crop padding (default: 1.3)'
    )
    parser.add_argument(
        '--haar_scale', type=float, default=1.1,
        help='Haar cascade scaleFactor (default: 1.1)'
    )
    parser.add_argument(
        '--haar_min_neighbors', type=int, default=5,
        help='Haar cascade minNeighbors (default: 5)'
    )
    parser.add_argument(
        '--haar_min_size', type=int, default=60,
        help='Haar cascade minimum face size in pixels (default: 60)'
    )
    parser.add_argument(
        '--show_crops', action='store_true', default=False,
        help='Show a debug window with the color/depth/IR face crops fed to the model'
    )
    parser.add_argument(
        '--depth_validity_override', type=float, default=0.0,
        help='If > 0, boost spoof probability when depth validity (fraction of valid '
             'depth pixels in face crop) is below this threshold. '
             'Phone screens cause near-zero depth validity. '
             'Recommended: 0.15. Default: 0 (disabled)'
    )
    parser.add_argument(
        '--headless', action='store_true', default=False,
        help='Run without display windows — prints predictions to terminal. '
             'Use when running over SSH without X11 forwarding.'
    )
    parser.add_argument(
        '--trt_engine', type=str, default=None,
        help='Path to a TensorRT engine file (.trt / .engine). '
             'When provided, uses TRT for inference instead of PyTorch. '
             '--checkpoint is not required when this is set.'
    )
    parser.add_argument(
        '--trt_fp16', action='store_true', default=False,
        help='Engine was built with FP16 precision (input will be cast to float16).'
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Minimal config — same pattern as benchmark_fps.py, avoids importing
# config/config.py which would consume sys.argv at module level.
# ---------------------------------------------------------------------------

class MinimalConfig:
    """Minimal config object for get_model(). Mirrors benchmark_fps.py."""
    def __init__(self, model_name, is_multi=True,
                 guidance_modality='depth', adaptive_guidance=False):
        self.model = model_name
        self.is_Multi = is_multi
        self.guidance_modality = guidance_modality
        self.adaptive_guidance = adaptive_guidance


# ---------------------------------------------------------------------------
# Helpers (adapted from benchmark_fps.py)
# ---------------------------------------------------------------------------

def _infer_model_from_checkpoint(checkpoint_path):
    """Infer model name from checkpoint path segments."""
    for part in checkpoint_path.replace('\\', '/').split('/'):
        if 'ShffleNetV2' in part or 'ViT' in part:
            if '_Multi' in part:
                return part.split('_Multi')[0]
            elif '_Single' in part:
                return part.split('_Single')[0]
    return None


def _detect_adaptive_guidance(checkpoint_path, device):
    """Return True if checkpoint contains guidance_selector weights."""
    state_dict = torch.load(checkpoint_path, map_location=device)
    keys = list(state_dict.keys())
    if keys[0].startswith('module.'):
        keys = [k.replace('module.', '') for k in keys]
    return any('guidance_selector' in k for k in keys)


def _load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint, handling DataParallel prefix and old backbone key names."""
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Strip DataParallel 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    # Remap old named-backbone keys → ModuleList keys used by the current model:
    #   rgb_backbone.*   → backbones.0.*
    #   depth_backbone.* → backbones.1.*
    #   ir_backbone.*    → backbones.2.*
    remap = {
        'rgb_backbone.':   'backbones.0.',
        'depth_backbone.': 'backbones.1.',
        'ir_backbone.':    'backbones.2.',
    }
    remapped = {}
    for k, v in state_dict.items():
        for old, new in remap.items():
            if k.startswith(old):
                k = new + k[len(old):]
                break
        remapped[k] = v

    model.load_state_dict(remapped)
    return model


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(args, device):
    """Build model from config and load checkpoint weights."""
    # Add project root to sys.path so model imports resolve correctly
    # (this script lives one directory below the project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    from model.bulid_model import get_model  # noqa: E402 — intentional late import

    # Auto-infer model name from checkpoint path if not overridden by user
    inferred = _infer_model_from_checkpoint(args.checkpoint)
    if inferred and args.model == 'ShffleNetV2_hd_v1_hybrid_d':
        args.model = inferred
        print(f"Auto-inferred model: {args.model}")

    # Auto-detect adaptive guidance from checkpoint weights
    adaptive = _detect_adaptive_guidance(args.checkpoint, device)
    if adaptive and not args.adaptive_guidance:
        args.adaptive_guidance = True
        print("Auto-detected adaptive guidance from checkpoint")

    config = MinimalConfig(
        model_name=args.model,
        is_multi=args.is_multi,
        guidance_modality=args.guidance_modality,
        adaptive_guidance=args.adaptive_guidance,
    )

    net = get_model(config, num_class=2)
    net = _load_checkpoint(net, args.checkpoint, device)
    net.eval()
    net.to(device)

    mode_str = 'multi-modal (C+D+IR)' if args.is_multi else f'single-modal ({args.image_modality})'
    print(f"Model:    {args.model}")
    print(f"Mode:     {mode_str}")
    print(f"Device:   {device}")
    print(f"Adaptive: {args.adaptive_guidance}")
    return net


# ---------------------------------------------------------------------------
# Face detection
# ---------------------------------------------------------------------------

class FaceDetector:
    """Haar cascade face detector — no external model downloads needed."""

    def __init__(self, scale_factor=1.1, min_neighbors=5, min_size=60):
        # cv2.data not available in older OpenCV versions; search common paths
        cascade_filename = 'haarcascade_frontalface_default.xml'
        candidate_dirs = [
            getattr(getattr(cv2, 'data', None), 'haarcascades', None),
            '/usr/share/opencv4/haarcascades',
            '/usr/share/opencv/haarcascades',
            '/usr/local/share/opencv4/haarcascades',
            '/usr/local/share/OpenCV/haarcascades',
        ]
        cascade_path = None
        for d in candidate_dirs:
            if d and os.path.isfile(os.path.join(d, cascade_filename)):
                cascade_path = os.path.join(d, cascade_filename)
                break
        if cascade_path is None:
            raise RuntimeError(
                f"Could not find {cascade_filename}. "
                "Install opencv-data or set the path manually."
            )
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = (min_size, min_size)

    def detect(self, gray_frame):
        """Return list of (x, y, w, h) sorted by area descending (largest first).
        Runs on a half-resolution frame for ~4x speedup on Jetson Nano."""
        scale = 0.5
        small = cv2.resize(gray_frame, (0, 0), fx=scale, fy=scale)
        faces = self.detector.detectMultiScale(
            small,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        # Scale coordinates back to original resolution
        inv = 1.0 / scale
        faces = [(int(x*inv), int(y*inv), int(w*inv), int(h*inv)) for x, y, w, h in faces]
        return sorted(faces, key=lambda f: f[2] * f[3], reverse=True)


# ---------------------------------------------------------------------------
# Frame preprocessing
# ---------------------------------------------------------------------------

class FramePreprocessor:
    """Converts RealSense frames + face bbox into a model input tensor."""

    IMAGE_SIZE = 64

    def __init__(self, args):
        self.is_multi = args.is_multi
        self.image_modality = args.image_modality
        self.depth_min = args.depth_min
        self.depth_max = args.depth_max

    # --- helpers ---

    def _pad_and_crop(self, img, x, y, w, h, scale):
        """Expand bbox by scale, clamp to image bounds, return crop."""
        ih, iw = img.shape[:2]
        cx, cy = x + w // 2, y + h // 2
        half_w = int(w * scale / 2)
        half_h = int(h * scale / 2)
        x1 = max(0, cx - half_w)
        y1 = max(0, cy - half_h)
        x2 = min(iw, cx + half_w)
        y2 = min(ih, cy + half_h)
        return img[y1:y2, x1:x2]

    def _normalize_depth(self, depth_uint16):
        """
        uint16 (H,W) in mm → uint8 (H,W,3) grayscale replicated.
        Clips to [depth_min, depth_max] mm then scales to [0, 255].
        """
        d = depth_uint16.astype(np.float32)
        d = np.clip(d, self.depth_min, self.depth_max)
        denom = max(self.depth_max - self.depth_min, 1)
        d = (d - self.depth_min) / denom * 255.0
        d = d.astype(np.uint8)
        return np.stack([d, d, d], axis=-1)

    def _to_3ch_gray(self, gray_uint8):
        """uint8 (H,W) → uint8 (H,W,3) by replication."""
        return np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    def _to_tensor(self, hwc_uint8):
        """(H,W,C) uint8 → FloatTensor (1,C,H,W) normalized to [0,1]."""
        chw = np.transpose(hwc_uint8, (2, 0, 1)).astype(np.float32) / 255.0
        return torch.FloatTensor(chw).unsqueeze(0)

    # --- public ---

    def preprocess(self, color, depth, ir, x, y, w, h, scale=1.3):
        """
        Build model input tensor.

        Returns FloatTensor [1,12,H,W] (multi) or [1,3,H,W] (single),
        or None if the crop is degenerate.
        """
        try:
            if self.is_multi:
                return self._preprocess_multi(color, depth, ir, x, y, w, h, scale)
            else:
                return self._preprocess_single(color, depth, ir, x, y, w, h, scale)
        except Exception:
            return None

    def _preprocess_multi(self, color, depth, ir, x, y, w, h, scale):
        sz = self.IMAGE_SIZE

        # Color (BGR, 3ch)
        c_crop = self._pad_and_crop(color, x, y, w, h, scale)
        if c_crop.size == 0:
            return None
        color_64 = cv2.resize(c_crop, (sz, sz), interpolation=cv2.INTER_LINEAR)

        # Depth (uint16 mm → uint8 3ch)
        # Resize as float32 first with INTER_NEAREST to preserve mm values
        d_crop = self._pad_and_crop(depth, x, y, w, h, scale)
        if d_crop.size == 0:
            return None
        d_resized = cv2.resize(
            d_crop.astype(np.float32), (sz, sz), interpolation=cv2.INTER_NEAREST
        ).astype(np.uint16)
        depth_64 = self._normalize_depth(d_resized)

        # IR (uint8 grayscale → 3ch)
        ir_crop = self._pad_and_crop(ir, x, y, w, h, scale)
        if ir_crop.size == 0:
            return None
        ir_resized = cv2.resize(ir_crop, (sz, sz), interpolation=cv2.INTER_LINEAR)
        ir_64 = self._to_3ch_gray(ir_resized)

        # Thermal — RealSense has no thermal stream; channel is present in the
        # model's input but the network never actually processes channels [9:12],
        # so zeroing them is safe.
        thermal_64 = np.zeros((sz, sz, 3), dtype=np.uint8)

        # Concatenate: [color(3) | depth(3) | ir(3) | thermal(3)] → (64,64,12)
        stacked = np.concatenate([color_64, depth_64, ir_64, thermal_64], axis=2)
        return self._to_tensor(stacked)  # (1,12,64,64)

    def _preprocess_single(self, color, depth, ir, x, y, w, h, scale):
        sz = self.IMAGE_SIZE

        if self.image_modality == 'color':
            crop = self._pad_and_crop(color, x, y, w, h, scale)
            if crop.size == 0:
                return None
            img = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_LINEAR)
            # img is already (H,W,3) BGR

        elif self.image_modality == 'depth':
            crop = self._pad_and_crop(depth, x, y, w, h, scale)
            if crop.size == 0:
                return None
            resized = cv2.resize(
                crop.astype(np.float32), (sz, sz), interpolation=cv2.INTER_NEAREST
            ).astype(np.uint16)
            img = self._normalize_depth(resized)

        else:  # ir
            crop = self._pad_and_crop(ir, x, y, w, h, scale)
            if crop.size == 0:
                return None
            resized = cv2.resize(crop, (sz, sz), interpolation=cv2.INTER_LINEAR)
            img = self._to_3ch_gray(resized)

        return self._to_tensor(img)  # (1,3,64,64)


# ---------------------------------------------------------------------------
# RealSense capture
# ---------------------------------------------------------------------------

class RealSenseCapture:
    """
    Context manager for a pyrealsense2 pipeline.
    Provides aligned Color + Depth + IR frames.

    Usage:
        with RealSenseCapture() as cam:
            color, depth, ir = cam.get_frames()
    """

    FPS = 30
    W, H = 424, 240  # lower resolution for faster processing; 640x480 also works
    WARMUP_FRAMES = 15  # allow auto-exposure to settle

    def __init__(self):
        self._pipeline = None

    def __enter__(self):
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, self.W, self.H, rs.format.bgr8, self.FPS)
        cfg.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.FPS)
        # IR left sensor (index 1), 8-bit grayscale
        cfg.enable_stream(rs.stream.infrared, 1, self.W, self.H, rs.format.y8, self.FPS)

        self._pipeline.start(cfg)
        # Skip rs.align() — it's expensive and we downsample to 64x64 anyway;
        # small depth/IR spatial offset is negligible at inference resolution.

        print(f"RealSense started - {self.W}x{self.H} @ {self.FPS} fps")
        print(f"Warming up ({self.WARMUP_FRAMES} frames)...", end=' ', flush=True)
        for _ in range(self.WARMUP_FRAMES):
            self._pipeline.wait_for_frames()
        print("done.")
        return self

    def get_frames(self):
        """
        Returns (color, depth, ir) as numpy arrays.
            color : (H,W,3) uint8  BGR
            depth : (H,W)   uint16 mm
            ir    : (H,W)   uint8  grayscale
        Note: frames are not spatially aligned — acceptable at 64x64 inference resolution.
        """
        frames = self._pipeline.wait_for_frames(timeout_ms=5000)

        c = frames.get_color_frame()
        d = frames.get_depth_frame()
        i = frames.get_infrared_frame(1)

        if not c or not d or not i:
            raise RuntimeError("Incomplete frameset from RealSense")

        color = np.asanyarray(c.get_data())
        depth = np.asanyarray(d.get_data())
        ir    = np.asanyarray(i.get_data())
        return color, depth, ir

    def __exit__(self, *_):
        if self._pipeline is not None:
            self._pipeline.stop()
            print("RealSense pipeline stopped.")


# ---------------------------------------------------------------------------
# Prediction smoothing
# ---------------------------------------------------------------------------

class PredictionSmoother:
    """Rolling mean of P(spoof) over the last N frames."""

    def __init__(self, window_size=5):
        self._buf = collections.deque(maxlen=window_size)

    def update(self, spoof_prob):
        self._buf.append(spoof_prob)

    def get(self):
        return float(np.mean(self._buf)) if self._buf else 0.5

    def reset(self):
        self._buf.clear()


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_inference(net, tensor, device):
    """
    Forward pass → (spoof_prob, guidance_weights).

    spoof_prob      : float in [0, 1]
    guidance_weights: np.ndarray [3] (depth, color, ir) or None for single-modal
    """
    outputs = net(tensor.to(device))
    logit = outputs[0]                          # [1, 2]
    probs = F.softmax(logit, dim=1)
    spoof_prob = probs[0, 1].item()

    # Multi-modal hybrid model returns guidance_weights at index 5
    guidance_weights = None
    if len(outputs) >= 6 and outputs[5] is not None:
        gw = outputs[5]                         # [1, 3]
        guidance_weights = gw[0].cpu().numpy()  # (3,) — [depth, color, ir]

    return spoof_prob, guidance_weights


# ---------------------------------------------------------------------------
# TensorRT inference (optional fast path)
# ---------------------------------------------------------------------------

class TRTInferencer:
    """
    Wraps a TensorRT engine for fast inference.
    Engine must have been exported from this model via:
        benchmark_fps.py --export_onnx model.onnx
        trtexec --onnx=model.onnx --saveEngine=model.trt --fp16

    Expected outputs: logit, depth_feas, color_feas, ir_feas, x_map, guidance_weights
    """

    def __init__(self, engine_path, fp16=False):
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # initializes CUDA context
        except ImportError as e:
            raise RuntimeError(
                f"TensorRT/pycuda not available: {e}. "
                "Install with: pip3 install pycuda; TRT is pre-installed on Jetson."
            )

        self._cuda = cuda
        self._fp16 = fp16
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(engine_path, 'rb') as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())
        self._context = self._engine.create_execution_context()
        self._stream = cuda.Stream()

        # Allocate pinned host buffers and device buffers for all bindings
        self._in_bufs = []
        self._out_bufs = {}
        self._bindings = []

        for i in range(self._engine.num_bindings):
            name = self._engine.get_binding_name(i)
            shape = tuple(self._engine.get_binding_shape(i))
            np_dtype = trt.nptype(self._engine.get_binding_dtype(i))
            size = int(np.prod(shape))
            host = cuda.pagelocked_empty(size, np_dtype)
            dev  = cuda.mem_alloc(host.nbytes)
            self._bindings.append(int(dev))
            if self._engine.binding_is_input(i):
                self._in_bufs.append({'host': host, 'device': dev, 'shape': shape})
            else:
                self._out_bufs[name] = {'host': host, 'device': dev, 'shape': shape}

        print(f"TRT engine loaded: {os.path.basename(engine_path)}")
        print(f"TRT outputs: {list(self._out_bufs.keys())}")

    def infer(self, input_tensor):
        """
        input_tensor: torch.Tensor (1, 12, 64, 64) on any device
        Returns: (spoof_prob: float, guidance_weights: np.ndarray(3,) or None)
        """
        cuda = self._cuda
        inp = input_tensor.cpu().numpy()
        if self._fp16:
            inp = inp.astype(np.float16)
        else:
            inp = inp.astype(np.float32)

        np.copyto(self._in_bufs[0]['host'], inp.ravel())
        cuda.memcpy_htod_async(self._in_bufs[0]['device'], self._in_bufs[0]['host'], self._stream)
        self._context.execute_async_v2(bindings=self._bindings, stream_handle=self._stream.handle)
        for out in self._out_bufs.values():
            cuda.memcpy_dtoh_async(out['host'], out['device'], self._stream)
        self._stream.synchronize()

        logit = self._out_bufs['logit']['host'].reshape(self._out_bufs['logit']['shape'])
        exp_l = np.exp(logit.astype(np.float32) - logit.max())
        spoof_prob = float(exp_l[0, 1] / exp_l.sum())

        gw = None
        if 'guidance_weights' in self._out_bufs:
            gw = self._out_bufs['guidance_weights']['host'].reshape(
                self._out_bufs['guidance_weights']['shape'])[0].astype(np.float32)
        return spoof_prob, gw


# ---------------------------------------------------------------------------
# Depth colormap helper
# ---------------------------------------------------------------------------

def depth_validity(depth_crop_raw, depth_min, depth_max):
    """
    Fraction of pixels in a raw uint16 depth crop that have valid readings,
    i.e., depth in [depth_min, depth_max] mm (non-zero, within sensor range).

    Real face:    typically 0.5 – 0.9  (most pixels hit the face)
    Phone screen: typically 0.0 – 0.05 (glossy screen reflects IR → invalid depth)
    """
    valid = (depth_crop_raw >= depth_min) & (depth_crop_raw <= depth_max)
    return float(valid.sum()) / max(depth_crop_raw.size, 1)


def make_crops_debug(preprocessor, color, depth, ir, x, y, w, h, scale, tile_size=128):
    """
    Build a side-by-side BGR image of the color/depth/IR crops fed to the model.
    Each tile is tile_size×tile_size for visibility (upscaled from 64×64).
    Also annotates the depth tile with the validity percentage.
    Returns (image, validity_fraction) or (None, None) if crop is degenerate.
    """
    sz = FramePreprocessor.IMAGE_SIZE
    pad = preprocessor._pad_and_crop

    # Color crop
    c = pad(color, x, y, w, h, scale)
    if c.size == 0:
        return None, None
    c64 = cv2.resize(c, (sz, sz))

    # Depth crop → normalized uint8 3ch
    d = pad(depth, x, y, w, h, scale)
    if d.size == 0:
        return None, None
    d64_raw = cv2.resize(d.astype(np.float32), (sz, sz),
                         interpolation=cv2.INTER_NEAREST).astype(np.uint16)
    validity = depth_validity(d64_raw, preprocessor.depth_min, preprocessor.depth_max)
    d64 = preprocessor._normalize_depth(d64_raw)

    # IR crop → 3ch gray
    i = pad(ir, x, y, w, h, scale)
    if i.size == 0:
        return None, None
    i64 = preprocessor._to_3ch_gray(cv2.resize(i, (sz, sz)))

    # Upscale each tile and label
    tiles = []
    for tile, label in [(c64, 'Color'), (d64, 'Depth'), (i64, 'IR')]:
        big = cv2.resize(tile, (tile_size, tile_size), interpolation=cv2.INTER_NEAREST)
        cv2.putText(big, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 0), 1, cv2.LINE_AA)
        tiles.append(big)

    # Annotate depth tile with validity score
    val_color = (0, 200, 0) if validity > 0.3 else (0, 0, 220)
    cv2.putText(tiles[1], f'valid:{validity*100:.0f}%',
                (4, tile_size - 6), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, val_color, 1, cv2.LINE_AA)

    return np.concatenate(tiles, axis=1), validity   # (tile_size, 3*tile_size, 3)


def make_depth_colormap(depth_raw, depth_min, depth_max):
    """uint16 depth array → colorized BGR image (COLORMAP_JET, near=warm)."""
    d = depth_raw.astype(np.float32)
    d = np.clip(d, depth_min, depth_max)
    denom = max(depth_max - depth_min, 1)
    d_norm = ((d - depth_min) / denom * 255).astype(np.uint8)
    d_inv = 255 - d_norm          # invert so near objects are warm/bright
    return cv2.applyColorMap(d_inv, cv2.COLORMAP_JET)


# ---------------------------------------------------------------------------
# Overlay rendering
# ---------------------------------------------------------------------------

class OverlayRenderer:
    """Draws bounding box, label, confidence bar, and FPS onto a color frame."""

    FONT      = cv2.FONT_HERSHEY_SIMPLEX
    C_REAL    = (0,   200,   0)    # green
    C_SPOOF   = (0,     0, 220)    # red
    C_NEUTRAL = (180, 180, 180)    # grey
    C_BAR_BG  = (50,   50,  50)
    C_TEXT    = (255, 255, 255)
    THRESHOLD = 0.5

    def draw(self, frame, face_bbox, smoothed_prob, fps, modality_label='',
             guidance_weights=None, depth_val=None):
        h, w = frame.shape[:2]

        # FPS — top left
        cv2.putText(frame, f'FPS: {fps:.1f}',
                    (10, 30), self.FONT, 0.8, self.C_TEXT, 2, cv2.LINE_AA)

        # Modality — top right
        if modality_label:
            text_size = cv2.getTextSize(modality_label, self.FONT, 0.65, 1)[0]
            cv2.putText(frame, modality_label,
                        (w - text_size[0] - 10, 30),
                        self.FONT, 0.65, self.C_TEXT, 1, cv2.LINE_AA)

        if face_bbox is None:
            cv2.putText(frame, 'No face detected',
                        (10, h - 55), self.FONT, 0.7, self.C_NEUTRAL, 2, cv2.LINE_AA)
            self._draw_confidence_bar(frame, smoothed_prob, w, h)
            return

        x, y, bw, bh = face_bbox
        is_spoof = smoothed_prob > self.THRESHOLD
        color    = self.C_SPOOF if is_spoof else self.C_REAL
        label    = 'SPOOF' if is_spoof else 'REAL'
        conf     = smoothed_prob if is_spoof else (1.0 - smoothed_prob)

        # Bounding box
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

        # Label above box
        text = f'{label}  {conf * 100:.1f}%'
        ty = max(y - 10, 25)
        cv2.putText(frame, text, (x, ty), self.FONT, 0.85, color, 2, cv2.LINE_AA)

        # Guidance weights + depth validity below bounding box
        info_y = min(y + bh + 18, h - 50)
        if guidance_weights is not None:
            d, c, i = guidance_weights
            gw_text = f'D:{d*100:.0f}%  C:{c*100:.0f}%  IR:{i*100:.0f}%'
            cv2.putText(frame, gw_text,
                        (x, info_y), self.FONT, 0.55, (200, 200, 0), 1, cv2.LINE_AA)
            info_y += 18
        if depth_val is not None and info_y < h - 50:
            dv_color = (0, 200, 0) if depth_val > 0.3 else (0, 0, 220)
            cv2.putText(frame, f'depth valid: {depth_val*100:.0f}%',
                        (x, info_y), self.FONT, 0.55, dv_color, 1, cv2.LINE_AA)

        self._draw_confidence_bar(frame, smoothed_prob, w, h)

    def _draw_confidence_bar(self, frame, prob, w, h):
        """Horizontal bar at bottom: left=REAL (green), right=SPOOF (red)."""
        bx1, by1 = 10, h - 40
        bx2, by2 = w - 10, h - 15
        bar_w = bx2 - bx1

        # Background
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), self.C_BAR_BG, -1)

        # Spoof fill (grows from right when prob rises)
        fill_w = int(bar_w * prob)
        if fill_w > 0:
            cv2.rectangle(frame, (bx1, by1), (bx1 + fill_w, by2),
                          self.C_SPOOF, -1)

        # Midpoint divider
        mid = bx1 + bar_w // 2
        cv2.line(frame, (mid, by1), (mid, by2), self.C_TEXT, 1)

        # Labels
        cv2.putText(frame, 'REAL',  (bx1 + 4, by2 - 4),
                    self.FONT, 0.45, self.C_REAL,  1, cv2.LINE_AA)
        cv2.putText(frame, 'SPOOF', (bx2 - 58, by2 - 4),
                    self.FONT, 0.45, self.C_SPOOF, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

class RealtimeFASDemo:
    WINDOW_MAIN  = 'FAS Live Demo - Q:quit  R:reset smoother'
    WINDOW_DEPTH = 'Depth Map'
    WINDOW_CROPS = 'Face Crops (Color | Depth | IR)'

    def __init__(self, args):
        self.args = args

        if args.device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(args.device)

        trt_path = getattr(args, 'trt_engine', None)
        if trt_path:
            self.net = None
            self.trt = TRTInferencer(trt_path, fp16=getattr(args, 'trt_fp16', False))
        else:
            self.net = load_model(args, self.device)
            self.trt = None
        self.detector     = FaceDetector(args.haar_scale,
                                         args.haar_min_neighbors,
                                         args.haar_min_size)
        self.preprocessor = FramePreprocessor(args)
        self.smoother     = PredictionSmoother(args.smooth_window)
        self.renderer     = OverlayRenderer()

        self._fps_buf  = collections.deque(maxlen=30)
        self._prev_t   = None

        self._mode_label = ('multi (C+D+IR)' if args.is_multi
                            else f'single ({args.image_modality})')

    def _fps(self):
        now = time.perf_counter()
        if self._prev_t is not None:
            dt = now - self._prev_t
            if dt > 0:
                self._fps_buf.append(1.0 / dt)
        self._prev_t = now
        return float(np.mean(self._fps_buf)) if self._fps_buf else 0.0

    def run(self):
        headless = getattr(self.args, 'headless', False)
        if headless:
            print("\nHeadless mode - press Ctrl+C to quit.\n")
        else:
            print("\nPress  Q  to quit,  R  to reset the prediction smoother.\n")
        try:
            with RealSenseCapture() as cam:
                if not headless:
                    cv2.namedWindow(self.WINDOW_MAIN, cv2.WINDOW_NORMAL)
                    if self.args.show_depth:
                        cv2.namedWindow(self.WINDOW_DEPTH, cv2.WINDOW_NORMAL)
                    if self.args.show_crops:
                        cv2.namedWindow(self.WINDOW_CROPS, cv2.WINDOW_NORMAL)

                last_guidance = None
                last_depth_val = None
                frame_count = 0
                last_faces = []
                DETECT_EVERY = 4  # run Haar only every N frames; reuse bbox in between

                while True:
                    # Capture
                    try:
                        color, depth, ir = cam.get_frames()
                    except RuntimeError as e:
                        print(f"Frame error: {e}")
                        continue

                    # Face detection: only every DETECT_EVERY frames to save CPU
                    if frame_count % DETECT_EVERY == 0:
                        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                        last_faces = self.detector.detect(gray)
                    faces = last_faces

                    display      = color.copy()
                    bbox_to_draw = None

                    if faces:
                        x, y, w, h = faces[0]   # largest face
                        bbox_to_draw = (x, y, w, h)

                        tensor = self.preprocessor.preprocess(
                            color, depth, ir, x, y, w, h,
                            scale=self.args.face_scale,
                        )
                        # Depth validity check (before model inference)
                        d_crop = self.preprocessor._pad_and_crop(depth, x, y, w, h,
                                                                  self.args.face_scale)
                        val = depth_validity(d_crop,
                                             self.args.depth_min, self.args.depth_max)
                        last_depth_val = val

                        if tensor is not None:
                            if self.trt is not None:
                                prob, gw = self.trt.infer(tensor)
                            else:
                                prob, gw = run_inference(self.net, tensor, self.device)

                            # Depth validity override: phone screens cause near-zero
                            # valid depth because the glossy surface reflects IR.
                            # When enabled, clamp probability toward spoof if validity
                            # is suspiciously low.
                            thresh = self.args.depth_validity_override
                            if thresh > 0 and val < thresh:
                                # Blend: weight toward 0.9 spoof the lower the validity
                                t = 1.0 - (val / thresh)   # 0→1 as validity→0
                                prob = prob + t * (0.9 - prob)

                            self.smoother.update(prob)
                            if gw is not None:
                                last_guidance = gw

                        # Show debug crops
                        if not headless and self.args.show_crops:
                            crops, _ = make_crops_debug(
                                self.preprocessor, color, depth, ir,
                                x, y, w, h, self.args.face_scale,
                            )
                            if crops is not None:
                                cv2.imshow(self.WINDOW_CROPS, crops)

                    fps = self._fps()
                    smooth_prob = self.smoother.get()
                    label = 'REAL' if smooth_prob < 0.5 else 'SPOOF'

                    if headless:
                        frame_count += 1
                        if frame_count % 10 == 0:  # print every 10 frames
                            face_str = f"face@({x},{y})" if faces else "no face"
                            print(f"[{frame_count:5d}] {label} ({smooth_prob:.2f})  "
                                  f"fps={fps:.1f}  {face_str}")
                    else:
                        # Render
                        self.renderer.draw(
                            display,
                            bbox_to_draw,
                            smooth_prob,
                            fps,
                            modality_label=self._mode_label,
                            guidance_weights=last_guidance,
                            depth_val=last_depth_val,
                        )
                        cv2.imshow(self.WINDOW_MAIN, display)

                        if self.args.show_depth:
                            depth_vis = make_depth_colormap(
                                depth, self.args.depth_min, self.args.depth_max
                            )
                            cv2.imshow(self.WINDOW_DEPTH, depth_vis)

                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord('q'), ord('Q')):
                            print("Quit.")
                            break
                        elif key in (ord('r'), ord('R')):
                            self.smoother.reset()
                            print("Smoother reset.")

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            if not headless:
                cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    trt_path = getattr(args, 'trt_engine', None)
    if trt_path:
        if not os.path.isfile(trt_path):
            print(f"ERROR: TRT engine not found: {trt_path}")
            sys.exit(1)
    elif args.checkpoint:
        if not os.path.isfile(args.checkpoint):
            print(f"ERROR: checkpoint not found: {args.checkpoint}")
            sys.exit(1)
    else:
        print("ERROR: must provide --checkpoint or --trt_engine")
        sys.exit(1)

    RealtimeFASDemo(args).run()


if __name__ == '__main__':
    main()
