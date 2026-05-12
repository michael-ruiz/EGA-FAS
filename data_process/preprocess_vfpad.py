#!/usr/bin/env python3
"""
Preprocess the VFPAD dataset: convert HDF5 face-frame files to JPEG images and
generate standard two-column list files (relative_path label) for training.

HDF5 structure (actual format):
    FrameIndexes/0          (1,) |S16   → b'frame_0007'
    FrameIndexes/1          (1,) |S16   → b'frame_0022'
    ...
    Frame_frame_0007/array  (128,128) uint8   ← already face-cropped NIR frame
    Frame_frame_0022/array  (128,128) uint8
    ...

Each HDF5 file = one video recording; frames are already selected and cropped.
Typical counts: ~20 frames/video (bf) or ~80 frames/video (pa).

Usage:
    python data_process/preprocess_vfpad.py [--frame_interval 1] [--vfpad_root PATH]

Outputs (relative to vfpad_root):
    frames/{rel_path}/{frame_name}.jpg
    protocol/grandtest/train_list.txt
    protocol/grandtest/dev_list.txt
    protocol/grandtest/eval_list.txt
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

def get_frame_names(f):
    """
    Return ordered list of short frame-name strings.
    e.g. ['frame_0007', 'frame_0022', ...]

    FrameIndexes stores the full group name 'Frame_frame_0007'; we strip the
    leading 'Frame_' so callers can reconstruct the key as 'Frame_{name}/array'.
    Falls back to scanning Frame_frame_* keys if FrameIndexes is absent.
    """
    if 'FrameIndexes' in f:
        names = []
        i = 0
        fi = f['FrameIndexes']
        while str(i) in fi:
            val = fi[str(i)][0]
            name = val.decode('utf-8') if isinstance(val, bytes) else str(val)
            name = name.strip()
            # Strip the 'Frame_' group prefix if present → 'frame_0007'
            if name.startswith('Frame_'):
                name = name[len('Frame_'):]
            names.append(name)
            i += 1
        return names

    # Fallback: collect Frame_frame_* keys and sort by frame number
    names = []
    for key in f.keys():
        if key.startswith('Frame_frame_'):
            # 'Frame_frame_0007' → 'frame_0007'
            names.append(key[len('Frame_'):])
    names.sort()
    return names


def process_hdf5(hdf5_path, output_dir, frame_interval):
    """
    Extract frames from one HDF5 file and save as 3-channel JPEG.
    Returns sorted list of saved filenames (basename only).
    """
    saved = []
    try:
        with h5py.File(hdf5_path, 'r') as f:
            frame_names = get_frame_names(f)
            if not frame_names:
                print(f'  [WARN] No frames found in {hdf5_path}')
                return saved

            os.makedirs(output_dir, exist_ok=True)

            for i, frame_name in enumerate(frame_names):
                if i % frame_interval != 0:
                    continue

                key = f'Frame_{frame_name}/array'
                if key not in f:
                    continue

                img = np.array(f[key])           # (128,128) uint8 grayscale
                img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

                fname = f'{frame_name}.jpg'
                cv2.imwrite(os.path.join(output_dir, fname), img_bgr,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append(fname)

    except Exception as e:
        print(f'  [ERROR] {hdf5_path}: {e}')

    return saved


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split, vfpad_root, frame_interval):
    """
    Process one split and return (rel_path, label) pairs
    where rel_path is relative to vfpad_root.
    """
    prot_dir    = os.path.join(vfpad_root, 'protocol', 'grandtest', split)
    data_dir    = os.path.join(vfpad_root, 'data')
    frames_root = os.path.join(vfpad_root, 'frames')

    entries = []

    for lst_file, label in [('for_real.lst', 1), ('for_attack.lst', 0)]:
        lst_path = os.path.join(prot_dir, lst_file)
        if not os.path.exists(lst_path):
            print(f'  [WARN] {lst_path} not found, skipping')
            continue

        with open(lst_path) as fh:
            lines = [ln.strip() for ln in fh if ln.strip()]

        print(f'\n  {split}/{lst_file}  ({len(lines)} videos, label={label})')

        for line in tqdm(lines, desc=f'{split}/{lst_file}', unit='vid'):
            rel_path = line.split()[0]   # e.g. bf/0006/bf_03_1_1_0006_...

            hdf5_path  = os.path.join(data_dir,    rel_path + '.hdf5')
            output_dir = os.path.join(frames_root, rel_path)

            if not os.path.exists(hdf5_path):
                print(f'  [WARN] HDF5 not found: {hdf5_path}')
                continue

            # Resume: reuse already-extracted frames
            if os.path.isdir(output_dir):
                existing = sorted(f for f in os.listdir(output_dir)
                                  if f.endswith('.jpg'))
                if existing:
                    for fname in existing:
                        entries.append((f'frames/{rel_path}/{fname}', label))
                    continue

            saved = process_hdf5(hdf5_path, output_dir, frame_interval)
            for fname in saved:
                entries.append((f'frames/{rel_path}/{fname}', label))

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess VFPAD: HDF5 face frames → JPEG + list files')
    parser.add_argument('--vfpad_root', type=str, default=None,
                        help='Path to VFPAD root dir (default: auto-detect)')
    parser.add_argument('--frame_interval', type=int, default=1,
                        help='Use every Nth pre-extracted frame per video (default: 1 = all)')
    parser.add_argument('--debug', action='store_true',
                        help='Print HDF5 structure of first file and exit')
    args = parser.parse_args()

    # Resolve vfpad_root
    if args.vfpad_root is None:
        script_dir   = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        vfpad_root   = os.path.join(project_root, 'datasets', 'VFPAD')
    else:
        vfpad_root = args.vfpad_root

    if not os.path.isdir(vfpad_root):
        raise FileNotFoundError(f'VFPAD root not found: {vfpad_root}')

    print(f'VFPAD root    : {vfpad_root}')
    print(f'Frame interval: every {args.frame_interval} available frame(s)')

    # --debug: inspect first HDF5 and exit
    if args.debug:
        import glob
        hdf5s = glob.glob(os.path.join(vfpad_root, 'data', '**', '*.hdf5'),
                          recursive=True)
        if not hdf5s:
            print('No HDF5 files found.')
            return
        print(f'\nStructure of {hdf5s[0]}:')
        with h5py.File(hdf5s[0], 'r') as f:
            def _print(name, obj):
                shape = obj.shape if isinstance(obj, h5py.Dataset) else ''
                dtype = obj.dtype if isinstance(obj, h5py.Dataset) else ''
                print(f'  {name:60s} {str(shape):20s} {dtype}')
            f.visititems(_print)
            names = get_frame_names(f)
            print(f'\nFrame names ({len(names)} total): {names[:5]} ...')
        return

    for split in ['train', 'dev', 'eval']:
        print(f'\n{"="*60}')
        print(f'Split: {split}')
        print('='*60)

        entries = process_split(split, vfpad_root, args.frame_interval)

        out_path = os.path.join(vfpad_root, 'protocol', 'grandtest',
                                f'{split}_list.txt')
        with open(out_path, 'w') as fh:
            for rel_path, label in entries:
                fh.write(f'{rel_path} {label}\n')

        real_n  = sum(1 for _, l in entries if l == 1)
        spoof_n = sum(1 for _, l in entries if l == 0)
        print(f'\n  Wrote {len(entries)} entries → {out_path}')
        print(f'  Real: {real_n}   Spoof: {spoof_n}')

    print('\nPreprocessing complete.')


if __name__ == '__main__':
    main()
