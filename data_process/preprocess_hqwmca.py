#!/usr/bin/env python3
"""
Preprocess the HQ-WMCA dataset: extract 3 spectral-group images from
43-channel SWIR-difference HDF5 frames and generate 6-column WMCA-format
list files for multi-modal training.

HDF5 MC-PixBiS-224 format:
    Per-frame array: (43, 224, 224) uint8
    Channel 0:   Grayscale (visible light)
    Channels 1-42: 21 SWIR normalized-difference pairs from 7 wavelengths
                   (940, 1050, 1200, 1300, 1450, 1550, 1650 nm)
                   d(λ1,λ2) = (I_λ1 - I_λ2)/(I_λ1 + I_λ2 + ε)  →  uint8

    Channels come in complementary pairs (Ch_i + Ch_{i+1} ≈ 255).
    Most discriminative: channels involving 1450/1550 nm (water absorption).

Default channel groups (for 3-backbone multi-modal model):
    Group 1 ("color"): Ch0  (visible), Ch1  (940-1050), Ch5  (940-1300)
    Group 2 ("depth"): Ch9  (940-1550), Ch27 (1200-1550), Ch33 (1300-1550)
    Group 3 ("ir"):    Ch7  (940-1450), Ch25 (1200-1450), Ch31 (1300-1450)

Input protocol lists (2-column):
    datasets/HQ-WMCA/protocols/{prot}/train_list.txt
    Format: relative/path/to/file.hdf5 label

Output:
    datasets/HQ-WMCA/frames/{video_stem}/
        color_frame_{idx:04d}.jpg   (group1: 3-ch image)
        depth_frame_{idx:04d}.jpg   (group2: 3-ch image)
        ir_frame_{idx:04d}.jpg      (group3: 3-ch image)
        thermal_frame_{idx:04d}.jpg (duplicate of group1 for 6-col compat)
    datasets/HQ-WMCA/protocols/{prot}/
        train_list_multi.txt   (6-column WMCA format)
        val_list_multi.txt
        test_list_multi.txt

Usage:
    python data_process/preprocess_hqwmca.py [--frame_interval 5] [--hqwmca_root PATH]
    python data_process/preprocess_hqwmca.py --debug
"""
import os
import argparse
import numpy as np
import cv2
import h5py
from tqdm import tqdm


# ---------------------------------------------------------------------------
# HDF5 reading
# ---------------------------------------------------------------------------

def get_frame_keys(f):
    """
    Get ordered list of frame group keys from HQ-WMCA HDF5.
    Structure: FrameIndexes/0 -> b'6', Frame_6/array -> (43, 224, 224)
    Returns list of frame name strings (e.g. ['6', '8', '9']).
    """
    if 'FrameIndexes' in f:
        names = []
        i = 0
        fi = f['FrameIndexes']
        while str(i) in fi:
            val = fi[str(i)][0]
            name = val.decode('utf-8') if isinstance(val, bytes) else str(val)
            name = name.strip()
            names.append(name)
            i += 1
        return names

    # Fallback: scan Frame_* keys (keep full group name)
    names = []
    for key in f.keys():
        if key.startswith('Frame_') and key != 'FrameIndexes':
            names.append(key)
    names.sort()
    return names


def debug_hdf5(hdf5_path):
    """Print HDF5 structure and display sample frames for debugging."""
    print(f'\nStructure of {hdf5_path}:')
    with h5py.File(hdf5_path, 'r') as f:
        def _print(name, obj):
            shape = obj.shape if isinstance(obj, h5py.Dataset) else ''
            dtype = obj.dtype if isinstance(obj, h5py.Dataset) else ''
            print(f'  {name:60s} {str(shape):20s} {dtype}')
        f.visititems(_print)

        frame_keys = get_frame_keys(f)
        print(f'\nFrame keys ({len(frame_keys)} total): {frame_keys[:5]} ...')

        if frame_keys:
            key = f'Frame_{frame_keys[0]}/array'
            if key in f:
                arr = np.array(f[key])  # (43, H, W)
                print(f'\nSample frame "{frame_keys[0]}" shape: {arr.shape}')
                print(f'  dtype: {arr.dtype}, range: [{arr.min()}, {arr.max()}]')
                for ch in range(min(arr.shape[0], 10)):
                    ch_data = arr[ch]
                    print(f'  Channel {ch:2d}: min={ch_data.min():8.2f}  '
                          f'max={ch_data.max():8.2f}  '
                          f'mean={ch_data.mean():8.2f}')


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def normalize_to_uint8(arr):
    """
    Normalize a single-channel array to uint8 [0, 255].
    If already uint8, return as-is. Otherwise min-max normalize.
    """
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    vmin, vmax = arr.min(), arr.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8)


def extract_modality_frames(hdf5_path, output_dir, frame_interval,
                            group1_channels, group2_channels,
                            group3_channels):
    """
    Extract 3 spectral-group frames from one HDF5 file and save as JPEG.
    Each group is a list of 3 channel indices -> stacked into a 3-ch image.
    HDF5 per-frame structure: Frame_{name}/array -> (43, H, W) uint8
    Returns list of (out_idx, color_fname, depth_fname, ir_fname, thermal_fname).
    """
    saved = []
    try:
        with h5py.File(hdf5_path, 'r') as f:
            frame_keys = get_frame_keys(f)
            if not frame_keys:
                print(f'  [WARN] No frames in {hdf5_path}')
                return saved

            os.makedirs(output_dir, exist_ok=True)
            quality = [cv2.IMWRITE_JPEG_QUALITY, 95]

            all_channels = group1_channels + group2_channels + group3_channels
            max_ch = max(all_channels)

            out_idx = 0
            for i, fkey in enumerate(frame_keys):
                if i % frame_interval != 0:
                    continue

                arr_key = f'{fkey}/array'
                if arr_key not in f:
                    continue

                frame = np.array(f[arr_key])  # (43, H, W)
                num_channels = frame.shape[0]

                if max_ch >= num_channels:
                    print(f'  [WARN] Channel {max_ch} >= {num_channels} '
                          f'in {hdf5_path}')
                    return saved

                # Each group: stack 3 channels -> (H, W, 3)
                color_img = np.stack(
                    [normalize_to_uint8(frame[ch]) for ch in group1_channels],
                    axis=2)
                depth_img = np.stack(
                    [normalize_to_uint8(frame[ch]) for ch in group2_channels],
                    axis=2)
                ir_img = np.stack(
                    [normalize_to_uint8(frame[ch]) for ch in group3_channels],
                    axis=2)
                # Thermal = duplicate of group1 for 6-column compatibility
                thermal_img = color_img.copy()

                color_fname   = f'color_frame_{out_idx:04d}.jpg'
                depth_fname   = f'depth_frame_{out_idx:04d}.jpg'
                ir_fname      = f'ir_frame_{out_idx:04d}.jpg'
                thermal_fname = f'thermal_frame_{out_idx:04d}.jpg'

                cv2.imwrite(os.path.join(output_dir, color_fname),
                            color_img, quality)
                cv2.imwrite(os.path.join(output_dir, depth_fname),
                            depth_img, quality)
                cv2.imwrite(os.path.join(output_dir, ir_fname),
                            ir_img, quality)
                cv2.imwrite(os.path.join(output_dir, thermal_fname),
                            thermal_img, quality)

                saved.append((out_idx, color_fname, depth_fname,
                              ir_fname, thermal_fname))
                out_idx += 1

    except Exception as e:
        print(f'  [ERROR] {hdf5_path}: {e}')

    return saved


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def hdf5_path_to_stem(rel_path):
    """
    Convert a relative HDF5 path to a flat video stem for the frames directory.
    e.g. 'MC-PixBiS-224/preprocessed/face-station/foo/bar.hdf5'
         -> 'MC-PixBiS-224_preprocessed_face-station_foo_bar'
    """
    stem = rel_path
    # Remove .hdf5 extension
    if stem.lower().endswith('.hdf5'):
        stem = stem[:-5]
    # Replace path separators with underscores
    stem = stem.replace('/', '_').replace('\\', '_')
    return stem


def process_split(split_name, list_path, hqwmca_root, frame_interval,
                  group1_channels, group2_channels, group3_channels,
                  frames_dir='frames'):
    """
    Process one protocol split list file.
    Returns list of 6-column entries:
        (color_path, color_path, depth_path, ir_path, thermal_path, label)
    """
    if not os.path.exists(list_path):
        print(f'  [WARN] {list_path} not found, skipping')
        return []

    with open(list_path) as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]

    if not lines:
        print(f'  [WARN] {list_path} is empty, skipping')
        return []

    print(f'\n  {split_name}: {len(lines)} entries')

    frames_root = os.path.join(hqwmca_root, frames_dir)
    entries = []

    for line in tqdm(lines, desc=split_name, unit='vid'):
        parts = line.split()
        if len(parts) < 2:
            print(f'  [WARN] Malformed line: {line}')
            continue

        rel_hdf5 = parts[0]
        # HQ-WMCA original convention: 0=bonafide, 1=attack
        # Flip to match WMCA convention: 1=real, 0=spoof
        label = 1 - int(parts[1])

        video_stem = hdf5_path_to_stem(rel_hdf5)
        output_dir = os.path.join(frames_root, video_stem)
        hdf5_path  = os.path.join(hqwmca_root, rel_hdf5)

        if not os.path.exists(hdf5_path):
            print(f'  [WARN] HDF5 not found: {hdf5_path}')
            continue

        # Resume: reuse already-extracted frames
        if os.path.isdir(output_dir):
            existing_jpgs = sorted(f for f in os.listdir(output_dir)
                                   if f.endswith('.jpg'))
            if existing_jpgs:
                # Reconstruct entries from existing files
                color_files = sorted(f for f in existing_jpgs
                                     if f.startswith('color_frame_'))
                for cf in color_files:
                    idx_str = cf.replace('color_frame_', '').replace('.jpg', '')
                    idx = idx_str  # keep as string for filename reconstruction
                    df = f'depth_frame_{idx}.jpg'
                    irf = f'ir_frame_{idx}.jpg'
                    tf = f'thermal_frame_{idx}.jpg'

                    # Verify all modalities exist
                    if all(os.path.exists(os.path.join(output_dir, f_))
                           for f_ in [df, irf, tf]):
                        c_rel = f'{frames_dir}/{video_stem}/{cf}'
                        d_rel = f'{frames_dir}/{video_stem}/{df}'
                        i_rel = f'{frames_dir}/{video_stem}/{irf}'
                        t_rel = f'{frames_dir}/{video_stem}/{tf}'
                        entries.append((c_rel, c_rel, d_rel, i_rel, t_rel, label))
                continue

        # Extract frames from HDF5
        saved = extract_modality_frames(
            hdf5_path, output_dir, frame_interval,
            group1_channels, group2_channels, group3_channels)

        for frame_idx, cf, df, irf, tf in saved:
            c_rel = f'{frames_dir}/{video_stem}/{cf}'
            d_rel = f'{frames_dir}/{video_stem}/{df}'
            i_rel = f'{frames_dir}/{video_stem}/{irf}'
            t_rel = f'{frames_dir}/{video_stem}/{tf}'
            entries.append((c_rel, c_rel, d_rel, i_rel, t_rel, label))

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess HQ-WMCA: 43-channel SWIR-diff HDF5 -> '
                    '3 spectral-group JPEG + 6-column WMCA-format list files')
    parser.add_argument('--hqwmca_root', type=str, default=None,
                        help='Path to HQ-WMCA root dir (default: auto-detect)')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Sample every Nth frame (default: 5)')
    parser.add_argument('--group1', type=str, default='0,1,5',
                        help='3 channel indices for group1 "color" slot '
                             '(default: 0,1,5 = visible + near-SWIR)')
    parser.add_argument('--group2', type=str, default='9,27,33',
                        help='3 channel indices for group2 "depth" slot '
                             '(default: 9,27,33 = 1550nm water absorption)')
    parser.add_argument('--group3', type=str, default='7,25,31',
                        help='3 channel indices for group3 "ir" slot '
                             '(default: 7,25,31 = 1450nm water absorption)')
    parser.add_argument('--frames_dir', type=str, default='frames_v2',
                        help='Output subdirectory for extracted frames '
                             '(default: frames_v2)')
    parser.add_argument('--debug', action='store_true',
                        help='Print HDF5 structure of first file and exit')
    args = parser.parse_args()

    # Parse channel groups
    group1_channels = [int(c.strip()) for c in args.group1.split(',')]
    group2_channels = [int(c.strip()) for c in args.group2.split(',')]
    group3_channels = [int(c.strip()) for c in args.group3.split(',')]
    assert len(group1_channels) == 3, 'group1 must have exactly 3 channels'
    assert len(group2_channels) == 3, 'group2 must have exactly 3 channels'
    assert len(group3_channels) == 3, 'group3 must have exactly 3 channels'

    # Resolve hqwmca_root
    if args.hqwmca_root is None:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        hqwmca_root  = os.path.join(project_root, 'datasets', 'HQ-WMCA')
    else:
        hqwmca_root = args.hqwmca_root

    if not os.path.isdir(hqwmca_root):
        raise FileNotFoundError(f'HQ-WMCA root not found: {hqwmca_root}')

    print(f'HQ-WMCA root    : {hqwmca_root}')
    print(f'Frame interval  : every {args.frame_interval} frame(s)')
    print(f'Group1 (color)  : {group1_channels}')
    print(f'Group2 (depth)  : {group2_channels}')
    print(f'Group3 (ir)     : {group3_channels}')
    print(f'Frames dir      : {args.frames_dir}')

    # --debug: inspect first HDF5 and exit
    if args.debug:
        import glob
        hdf5s = glob.glob(os.path.join(hqwmca_root, '**', '*.hdf5'),
                          recursive=True)
        if not hdf5s:
            print('No HDF5 files found.')
            return
        debug_hdf5(hdf5s[0])
        return

    # Discover all protocol directories
    protocols_root = os.path.join(hqwmca_root, 'protocols')
    if not os.path.isdir(protocols_root):
        raise FileNotFoundError(
            f'Protocols directory not found: {protocols_root}')

    prot_dirs = sorted([
        d for d in os.listdir(protocols_root)
        if os.path.isdir(os.path.join(protocols_root, d))
    ])

    if not prot_dirs:
        raise FileNotFoundError(
            f'No protocol subdirectories found in {protocols_root}')

    print(f'Protocols found : {prot_dirs}')

    # Split name mapping: input filename -> output filename
    split_map = {
        'train_list.txt': 'train_list_multi.txt',
        'val_list.txt':   'val_list_multi.txt',
        'test_list.txt':  'test_list_multi.txt',
    }

    for prot in prot_dirs:
        prot_dir = os.path.join(protocols_root, prot)

        # Check if this protocol has any list files
        available_splits = [s for s in split_map
                            if os.path.exists(os.path.join(prot_dir, s))]
        if not available_splits:
            print(f'\n  [SKIP] Protocol "{prot}" has no list files')
            continue

        print(f'\n{"="*60}')
        print(f'Protocol: {prot}')
        print('='*60)

        for input_name, output_name in split_map.items():
            list_path = os.path.join(prot_dir, input_name)
            if not os.path.exists(list_path):
                continue

            split_label = input_name.replace('_list.txt', '')
            entries = process_split(
                split_label, list_path, hqwmca_root, args.frame_interval,
                group1_channels, group2_channels, group3_channels,
                frames_dir=args.frames_dir)

            if not entries:
                print(f'  [WARN] No entries produced for {prot}/{input_name}')
                continue

            # Write 6-column output list
            out_path = os.path.join(prot_dir, output_name)
            with open(out_path, 'w') as fh:
                for col0, col1, col2, col3, col4, label in entries:
                    fh.write(f'{col0} {col1} {col2} {col3} {col4} {label}\n')

            real_n  = sum(1 for e in entries if e[5] == 1)
            spoof_n = sum(1 for e in entries if e[5] == 0)
            print(f'\n  Wrote {len(entries)} entries -> {out_path}')
            print(f'  Real: {real_n}   Spoof: {spoof_n}')

    print('\nPreprocessing complete.')


if __name__ == '__main__':
    main()
