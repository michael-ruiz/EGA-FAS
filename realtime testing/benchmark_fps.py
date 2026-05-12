#!/usr/bin/env python3
"""
FPS Benchmarking Script for LFAS-NewBackbone Models

Benchmarks model inference speed and estimates performance on Jetson AGX Xavier.

Usage:
    python benchmark_fps.py --model ShffleNetV2_hd_v1_hybrid_d
    python benchmark_fps.py --checkpoint path/to/model.pth
    python benchmark_fps.py --all_models
"""

import argparse
import time
import os
import numpy as np
import torch
from model.bulid_model import get_model


# GPU TFLOPS for scaling estimates (FP32)
GPU_TFLOPS = {
    'NVIDIA GeForce RTX 4090': 82.6,
    'NVIDIA GeForce RTX 4080': 48.7,
    'NVIDIA GeForce RTX 3090': 35.6,
    'NVIDIA GeForce RTX 3080': 29.8,
    'NVIDIA GeForce RTX 3070': 20.3,
    'NVIDIA GeForce RTX 2080 Ti': 13.4,
    'default': 30.0,  # Fallback estimate
}

# Jetson AGX Xavier specs
XAVIER_FP32_TFLOPS = 1.4
XAVIER_FP16_TFLOPS = 5.5


class MinimalConfig:
    """Minimal config object for get_model()"""
    def __init__(self, model_name, is_multi=True, guidance_modality='depth', adaptive_guidance=False):
        self.model = model_name
        self.is_Multi = is_multi
        self.guidance_modality = guidance_modality
        self.adaptive_guidance = adaptive_guidance


def get_gpu_tflops(device_name):
    """Get TFLOPS for the detected GPU"""
    for gpu_name, tflops in GPU_TFLOPS.items():
        if gpu_name in device_name:
            return tflops
    return GPU_TFLOPS['default']


def benchmark_model(model, input_tensor, device, warmup=50, iterations=500, sync=True):
    """
    Benchmark model inference speed.

    Returns:
        dict with latency and FPS statistics
    """
    model.eval()

    # Warm-up phase
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)
            if device.type == 'cuda' and sync:
                torch.cuda.synchronize()

    # Benchmark phase
    latencies = []
    with torch.inference_mode():
        for _ in range(iterations):
            if device.type == 'cuda' and sync:
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = model(input_tensor)

            if device.type == 'cuda' and sync:
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # Convert to ms

    latencies = np.array(latencies)
    fps_values = 1000.0 / latencies  # Convert latency (ms) to FPS

    return {
        'mean_latency_ms': np.mean(latencies),
        'std_latency_ms': np.std(latencies),
        'min_latency_ms': np.min(latencies),
        'max_latency_ms': np.max(latencies),
        'mean_fps': np.mean(fps_values),
        'min_fps': np.min(fps_values),
        'max_fps': np.max(fps_values),
    }


def estimate_xavier_fps(host_fps, host_tflops):
    """
    Estimate FPS on Jetson AGX Xavier based on TFLOPS ratio.

    Returns:
        tuple (fp32_fps, fp16_fps_estimate)
    """
    # FP32 scaling
    fp32_scale = XAVIER_FP32_TFLOPS / host_tflops
    fp32_fps = host_fps * fp32_scale

    # FP16/TensorRT estimate (assumes ~2x speedup from FP16 + TensorRT optimizations)
    fp16_scale = XAVIER_FP16_TFLOPS / host_tflops
    # TensorRT typically gives additional 1.5-2x speedup beyond raw FP16
    fp16_fps = host_fps * fp16_scale * 1.5

    return fp32_fps, fp16_fps


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_checkpoint(model, checkpoint_path, device):
    """Load checkpoint into model, handling DataParallel prefix and legacy key names"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Remove 'module.' prefix if model was saved with DataParallel
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    # Remap legacy named backbones to ModuleList format
    legacy_backbone_map = {
        'rgb_backbone.': 'backbones.0.',
        'depth_backbone.': 'backbones.1.',
        'ir_backbone.': 'backbones.2.',
    }
    remapped = {}
    for k, v in state_dict.items():
        for old_prefix, new_prefix in legacy_backbone_map.items():
            if k.startswith(old_prefix):
                k = new_prefix + k[len(old_prefix):]
                break
        remapped[k] = v
    state_dict = remapped
    model.load_state_dict(state_dict, strict=False)
    return model


def detect_adaptive_guidance(checkpoint_path, device):
    """Detect if checkpoint was trained with adaptive guidance"""
    state_dict = torch.load(checkpoint_path, map_location=device)
    # Remove 'module.' prefix if present
    keys = list(state_dict.keys())
    if keys[0].startswith('module.'):
        keys = [k.replace('module.', '') for k in keys]
    # Check for guidance_selector keys
    return any('guidance_selector' in k for k in keys)


def infer_model_from_checkpoint(checkpoint_path):
    """Infer model name from checkpoint path"""
    # Path typically contains model name like: .../ShffleNetV2_hd_v1_hybrid_d_Multi_64/...
    path_parts = checkpoint_path.split(os.sep)
    for part in path_parts:
        if 'ShffleNetV2' in part or 'ViT' in part:
            # Extract model name (before _Multi or _Single)
            if '_Multi' in part:
                return part.split('_Multi')[0]
            elif '_Single' in part:
                return part.split('_Single')[0]
    return None


def export_to_onnx(model, input_shape, output_path, device):
    """
    Export model to ONNX format and validate.

    Returns:
        dict with export status and info
    """
    import warnings

    result = {
        'success': False,
        'path': output_path,
        'warnings': [],
        'errors': [],
        'opset_version': 14,
    }

    # Create dummy input on CPU for export
    dummy_input = torch.randn(input_shape, device='cpu')
    model_cpu = model.cpu()
    model_cpu.eval()

    print(f"\nExporting to ONNX: {output_path}")
    print("-" * 40)

    # Export to ONNX using legacy JIT-based exporter (handles dynamic control flow better)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Use dynamo=False to force legacy JIT exporter (PyTorch 2.1+)
            export_kwargs = {
                'export_params': True,
                'opset_version': 12,  # Use opset 12 for better TensorRT compatibility
                'do_constant_folding': True,
                'input_names': ['input'],
                'output_names': ['logit', 'depth_feas', 'color_feas', 'ir_feas', 'x_map', 'guidance_weights'],
                'dynamic_axes': {
                    'input': {0: 'batch_size'},
                    'logit': {0: 'batch_size'},
                },
                'verbose': False,
            }
            # Try to use dynamo=False for PyTorch 2.1+
            try:
                torch.onnx.export(model_cpu, dummy_input, output_path, dynamo=False, **export_kwargs)
            except TypeError:
                # Older PyTorch without dynamo parameter
                torch.onnx.export(model_cpu, dummy_input, output_path, **export_kwargs)
            result['opset_version'] = 12
            # Collect warnings
            for warning in w:
                result['warnings'].append(str(warning.message))

        print(f"  Export successful (opset {result['opset_version']})")
        result['success'] = True

    except Exception as e:
        result['errors'].append(str(e))
        print(f"  Export FAILED: {e}")
        return result

    # Check file size
    if os.path.exists(output_path):
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        result['size_mb'] = size_mb
        print(f"  File size: {size_mb:.2f} MB")

    # Try to validate with onnx library
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("  ONNX validation: PASSED")
        result['validated'] = True

        # Count ops
        op_types = {}
        for node in onnx_model.graph.node:
            op_types[node.op_type] = op_types.get(node.op_type, 0) + 1
        result['op_counts'] = op_types
        print(f"  Total ops: {sum(op_types.values())}")
        print(f"  Op types: {len(op_types)}")

        # Check for potentially problematic ops for TensorRT
        problematic_ops = ['NonZero', 'Loop', 'If', 'Scan', 'SequenceAt', 'Where']
        found_problematic = [op for op in problematic_ops if op in op_types]
        if found_problematic:
            print(f"  WARNING: Ops that may not convert well to TensorRT: {found_problematic}")
            result['warnings'].append(f"Potentially problematic ops: {found_problematic}")
        else:
            print("  TensorRT compatibility: Likely OK (no known problematic ops)")

    except ImportError:
        print("  ONNX validation: SKIPPED (onnx package not installed)")
        result['validated'] = False
    except Exception as e:
        print(f"  ONNX validation: FAILED ({e})")
        result['errors'].append(f"Validation error: {e}")

    # Try to simplify with onnxsim
    try:
        import onnxsim
        simplified_path = output_path.replace('.onnx', '_simplified.onnx')
        onnx_model = onnx.load(output_path)
        model_simp, check = onnxsim.simplify(onnx_model)
        if check:
            onnx.save(model_simp, simplified_path)
            simp_size_mb = os.path.getsize(simplified_path) / (1024 * 1024)
            print(f"  Simplified model saved: {simplified_path} ({simp_size_mb:.2f} MB)")
            result['simplified_path'] = simplified_path
            result['simplified_size_mb'] = simp_size_mb
        else:
            print("  Simplification: FAILED (validation check failed)")
    except ImportError:
        print("  Simplification: SKIPPED (onnxsim not installed, run: pip install onnxsim)")
    except Exception as e:
        print(f"  Simplification: FAILED ({e})")

    # Move model back to original device
    model.to(device)

    # Analyze ONNX for performance estimation
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        analysis = analyze_onnx_performance(onnx_model, input_shape)
        result['analysis'] = analysis

        print(f"\n  Performance Analysis:")
        print(f"    Total FLOPs: {analysis['total_flops'] / 1e6:.2f} MFLOPs")
        print(f"    Total params: {analysis['total_params'] / 1e3:.1f} K")
        print(f"    ONNX ops: {analysis['num_ops']} -> ~{analysis['estimated_kernels']:.0f} kernels (after TRT fusion)")
        print(f"    Memory (weights): {analysis['memory_weights_mb']:.2f} MB")

        print(f"\n  Jetson AGX Xavier Estimates (from ONNX analysis):")
        print(f"    Estimated latency: {analysis['estimated_latency_ms']:.1f} ms")
        print(f"    Bottleneck: {analysis['bottleneck']}")
        print(f"    Theoretical FP16: ~{analysis['xavier_fp16_compute_fps']:.0f} FPS (if compute-bound)")
        print(f"    Latency-limited:  ~{analysis['xavier_latency_bound_fps']:.0f} FPS (kernel overhead)")
        print(f"    >>> Realistic estimate: ~{analysis['xavier_realistic_fps']:.0f} FPS (TensorRT FP16) <<<")

    except Exception as e:
        print(f"  Performance analysis failed: {e}")

    return result


def analyze_onnx_performance(onnx_model, input_shape):
    """
    Analyze ONNX model to estimate FLOPs and performance on different devices.
    """
    import onnx
    from onnx import numpy_helper

    total_flops = 0
    total_params = 0
    total_memory_bytes = 0

    # Get initializer shapes (weights)
    initializers = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

    for name, arr in initializers.items():
        total_params += arr.size
        total_memory_bytes += arr.nbytes

    # Estimate FLOPs from common operations
    # This is approximate - proper FLOPs counting requires shape inference
    for node in onnx_model.graph.node:
        op = node.op_type

        if op == 'Conv':
            # Try to get weight shape from initializers
            weight_name = node.input[1] if len(node.input) > 1 else None
            if weight_name and weight_name in initializers:
                w = initializers[weight_name]
                # Conv FLOPs = 2 * K * K * Cin * Cout * Hout * Wout
                # Approximate Hout/Wout from input shape
                out_channels, in_channels = w.shape[0], w.shape[1]
                kernel_size = w.shape[2] * w.shape[3] if len(w.shape) > 3 else w.shape[2]
                # Rough estimate assuming stride=2 reduces by half each conv
                spatial = (input_shape[2] * input_shape[3]) // 4  # rough average
                flops = 2 * kernel_size * in_channels * out_channels * spatial
                total_flops += flops

        elif op == 'Gemm' or op == 'MatMul':
            # Matrix multiply FLOPs
            for inp in node.input:
                if inp in initializers:
                    w = initializers[inp]
                    if len(w.shape) == 2:
                        total_flops += 2 * w.shape[0] * w.shape[1] * input_shape[0]

        elif op == 'BatchNormalization':
            # BN is relatively cheap: ~4 ops per element
            for inp in node.input[1:]:  # scale, bias, mean, var
                if inp in initializers:
                    total_flops += 4 * initializers[inp].size
                    break

    # Memory for activations (rough estimate: assume peak is 2x input size * channels)
    activation_memory = input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3] * 4 * 8  # 8x for intermediate

    # Xavier specs
    xavier_fp32_tflops = 1.4  # TFLOPS
    xavier_fp16_tflops = 5.5  # TFLOPS (with tensor cores)
    xavier_mem_bandwidth = 137  # GB/s

    # Count operations that will become CUDA kernels
    num_ops = len(list(onnx_model.graph.node))

    # Calculate theoretical limits
    # Compute-bound: FPS = TFLOPS / (FLOPs per inference)
    xavier_fp32_compute_fps = (xavier_fp32_tflops * 1e12) / max(total_flops, 1)
    xavier_fp16_compute_fps = (xavier_fp16_tflops * 1e12) / max(total_flops, 1)

    # Memory-bound: FPS = bandwidth / (bytes per inference)
    bytes_per_inference = total_memory_bytes + activation_memory
    xavier_membw_limit_fps = (xavier_mem_bandwidth * 1e9) / max(bytes_per_inference, 1)

    # Latency-bound estimate (critical for small models!)
    # TensorRT fuses ~3-5 ops per kernel on average
    # Xavier kernel launch overhead: ~10-20µs per kernel
    estimated_kernels = num_ops / 4  # TensorRT fusion estimate
    kernel_overhead_us = 15  # µs per kernel launch
    min_latency_ms = (estimated_kernels * kernel_overhead_us) / 1000
    # Add compute time
    compute_time_ms = (total_flops / (xavier_fp16_tflops * 1e12)) * 1000
    total_latency_ms = min_latency_ms + compute_time_ms
    xavier_latency_bound_fps = 1000 / max(total_latency_ms, 0.1)

    # Realistic estimate: limited by latency for small models
    # For small models (<500 MFLOPs), latency dominates
    # For large models (>1 GFLOPs), compute/memory dominates
    if total_flops < 500e6:  # Small model
        # Latency-bound with some efficiency loss
        realistic_fps = xavier_latency_bound_fps * 0.7
        bottleneck = "latency (kernel launches)"
    else:
        # Compute/memory bound
        theoretical_max = min(xavier_fp16_compute_fps, xavier_membw_limit_fps)
        realistic_fps = theoretical_max * 0.6
        bottleneck = "compute" if xavier_fp16_compute_fps < xavier_membw_limit_fps else "memory bandwidth"

    return {
        'total_flops': total_flops,
        'total_params': total_params,
        'num_ops': num_ops,
        'estimated_kernels': estimated_kernels,
        'memory_weights_mb': total_memory_bytes / (1024 * 1024),
        'memory_activations_mb': activation_memory / (1024 * 1024),
        'xavier_fp32_compute_fps': xavier_fp32_compute_fps,
        'xavier_fp16_compute_fps': xavier_fp16_compute_fps,
        'xavier_membw_limit_fps': xavier_membw_limit_fps,
        'xavier_latency_bound_fps': xavier_latency_bound_fps,
        'xavier_realistic_fps': realistic_fps,
        'bottleneck': bottleneck,
        'estimated_latency_ms': total_latency_ms,
    }


def main():
    parser = argparse.ArgumentParser(description='Benchmark FPS for LFAS models')
    parser.add_argument('--model', type=str, default='ShffleNetV2_hd_v1_hybrid_d',
                        help='Model name (default: ShffleNetV2_hd_v1_hybrid_d)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained checkpoint (.pth file)')
    parser.add_argument('--image_size', type=int, default=64,
                        help='Input image size (default: 64)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference (default: 1)')
    parser.add_argument('--warmup', type=int, default=50,
                        help='Warmup iterations (default: 50)')
    parser.add_argument('--iterations', type=int, default=500,
                        help='Benchmark iterations (default: 500)')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use (default: auto)')
    parser.add_argument('--all_models', action='store_true',
                        help='Benchmark all main model variants')
    parser.add_argument('--guidance_modality', type=str, default='depth',
                        choices=['depth', 'color', 'ir'],
                        help='Guidance modality (default: depth)')
    parser.add_argument('--adaptive_guidance', action='store_true',
                        help='Enable adaptive guidance')
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 (half precision) for inference')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for optimization (PyTorch 2.0+)')
    parser.add_argument('--no_sync', action='store_true',
                        help='Skip CUDA synchronization (may give inflated FPS)')
    parser.add_argument('--export_onnx', type=str, default=None, nargs='?', const='model.onnx',
                        help='Export model to ONNX format (optionally specify output path)')

    args = parser.parse_args()

    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Get GPU info
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_tflops = get_gpu_tflops(gpu_name)
    else:
        gpu_name = 'CPU'
        gpu_tflops = 1.0  # CPU baseline

    # Handle checkpoint mode
    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        # Try to infer model name from checkpoint path
        inferred_model = infer_model_from_checkpoint(checkpoint_path)
        if inferred_model:
            args.model = inferred_model
            print(f"Inferred model from checkpoint path: {args.model}")
        models_to_test = [args.model]
    elif args.all_models:
        models_to_test = [
            'ShffleNetV2_hd_v1',
            'ShffleNetV2_hd_v1_hybrid_a',
            'ShffleNetV2_hd_v1_hybrid_b',
            'ShffleNetV2_hd_v1_hybrid_c',
            'ShffleNetV2_hd_v1_hybrid_d',
            'ViT_hd_v1',
        ]
    else:
        models_to_test = [args.model]

    # Input tensor shape: [batch, 12, H, W] for multi-modal
    input_shape = (args.batch_size, 12, args.image_size, args.image_size)

    print("=" * 60)
    print("LFAS-NewBackbone FPS Benchmark")
    print("=" * 60)
    print(f"Device: {gpu_name}")
    if device.type == 'cuda':
        print(f"GPU TFLOPS (FP32): {gpu_tflops}")
    print(f"Input shape: {input_shape}")
    if checkpoint_path:
        print(f"Checkpoint: {checkpoint_path}")
    print(f"Warmup iterations: {args.warmup}")
    print(f"Benchmark iterations: {args.iterations}")
    opts = []
    if args.fp16:
        opts.append("FP16")
    if args.compile:
        opts.append("torch.compile")
    if args.no_sync:
        opts.append("no-sync")
    if opts:
        print(f"Optimizations: {', '.join(opts)}")
    print("=" * 60)

    results = []

    for model_name in models_to_test:
        print(f"\nBenchmarking: {model_name}")
        print("-" * 40)

        try:
            # Detect adaptive guidance from checkpoint if provided
            use_adaptive = args.adaptive_guidance
            if checkpoint_path:
                use_adaptive = detect_adaptive_guidance(checkpoint_path, device)
                if use_adaptive:
                    print("Detected adaptive guidance in checkpoint")

            # Create config and model
            config = MinimalConfig(
                model_name=model_name,
                is_multi=True,
                guidance_modality=args.guidance_modality,
                adaptive_guidance=use_adaptive
            )

            model = get_model(config, num_class=2)

            # Load checkpoint if provided
            if checkpoint_path:
                model = load_checkpoint(model, checkpoint_path, device)
                print(f"Loaded checkpoint: {os.path.basename(checkpoint_path)}")

            model = model.to(device)
            model.eval()

            # Count parameters
            total_params, trainable_params = count_parameters(model)
            print(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")

            # Export to ONNX if requested
            if args.export_onnx:
                onnx_path = args.export_onnx
                if not onnx_path.endswith('.onnx'):
                    onnx_path = f"{model_name}.onnx"
                onnx_result = export_to_onnx(model, input_shape, onnx_path, device)
                if not onnx_result['success']:
                    print("ONNX export failed, skipping benchmark")
                    continue

            # Apply FP16 if requested
            if args.fp16 and device.type == 'cuda':
                model = model.half()
                print("Using FP16 (half precision)")

            # Apply torch.compile if requested
            if args.compile:
                try:
                    model = torch.compile(model, mode='reduce-overhead')
                    print("Using torch.compile() with reduce-overhead mode")
                except Exception as e:
                    print(f"torch.compile() failed: {e}")

            # Create input tensor
            dtype = torch.float16 if (args.fp16 and device.type == 'cuda') else torch.float32
            input_tensor = torch.randn(input_shape, device=device, dtype=dtype)

            # Run benchmark
            stats = benchmark_model(
                model, input_tensor, device,
                warmup=args.warmup,
                iterations=args.iterations,
                sync=not args.no_sync
            )

            # Estimate Xavier performance
            xavier_fp32, xavier_fp16 = estimate_xavier_fps(stats['mean_fps'], gpu_tflops)

            # Print results
            print(f"\nResults ({args.iterations} iterations):")
            print(f"  Mean latency: {stats['mean_latency_ms']:.2f} ms (+/- {stats['std_latency_ms']:.2f})")
            print(f"  Min latency:  {stats['min_latency_ms']:.2f} ms")
            print(f"  Max latency:  {stats['max_latency_ms']:.2f} ms")
            print(f"  Mean FPS:     {stats['mean_fps']:.1f}")
            print(f"  FPS range:    {stats['min_fps']:.1f} - {stats['max_fps']:.1f}")

            if device.type == 'cuda':
                print(f"\nJetson AGX Xavier Estimates:")
                print(f"  FP32 mode:     ~{xavier_fp32:.1f} FPS")
                print(f"  FP16/TensorRT: ~{xavier_fp16:.1f} FPS (optimized)")

            results.append({
                'model': model_name,
                'params': total_params,
                'mean_fps': stats['mean_fps'],
                'mean_latency_ms': stats['mean_latency_ms'],
                'xavier_fp32': xavier_fp32 if device.type == 'cuda' else None,
                'xavier_fp16': xavier_fp16 if device.type == 'cuda' else None,
            })

            # Clean up
            del model, input_tensor
            if device.type == 'cuda':
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error benchmarking {model_name}: {e}")
            continue

    # Summary table
    if len(results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"{'Model':<35} {'Params':<10} {'FPS':<10} {'Xavier FP16':<12}")
        print("-" * 60)
        for r in results:
            xavier_str = f"~{r['xavier_fp16']:.0f}" if r['xavier_fp16'] else "N/A"
            print(f"{r['model']:<35} {r['params']/1e6:.2f}M     {r['mean_fps']:<10.1f} {xavier_str:<12}")


if __name__ == '__main__':
    main()
