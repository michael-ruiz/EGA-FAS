#!/usr/bin/env python3
"""
Extract visible-only (Ch0) images from HQ-WMCA frames_v2 color images
for single-modal cross-dataset experiments with WMCA.

HQ-WMCA Group1 (color slot) = [Ch0 visible, Ch1 SWIR, Ch5 SWIR]
This script extracts Ch0 and saves as 3-channel grayscale (replicated).

Also creates 2-column single-modal list files from existing 6-column lists.

Usage:
    python data_process/extract_hqwmca_visible.py
"""
import os
import cv2
import numpy as np
from tqdm import tqdm


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    hqwmca_root = os.path.join(project_root, 'datasets', 'HQ-WMCA')
    frames_v2 = os.path.join(hqwmca_root, 'frames_v2')
    frames_visible = os.path.join(hqwmca_root, 'frames_visible')

    if not os.path.isdir(frames_v2):
        raise FileNotFoundError(f'frames_v2 not found: {frames_v2}')

    # Step 1: Extract Ch0 from color_*.jpg files
    print('Extracting visible-only (Ch0) images from frames_v2...')
    color_files = []
    for root, dirs, files in os.walk(frames_v2):
        for f in files:
            if f.startswith('color_') and f.endswith('.jpg'):
                color_files.append(os.path.join(root, f))

    print(f'  Found {len(color_files)} color images')

    for src_path in tqdm(color_files, desc='Extracting Ch0', unit='img'):
        rel = os.path.relpath(src_path, frames_v2)
        # Change color_ prefix to visible_
        rel_dir = os.path.dirname(rel)
        fname = os.path.basename(rel).replace('color_', 'visible_')
        dst_path = os.path.join(frames_visible, rel_dir, fname)

        if os.path.exists(dst_path):
            continue

        img = cv2.imread(src_path)
        if img is None:
            continue

        # Ch0 is in the B channel (index 0) of the saved BGR image
        ch0 = img[:, :, 0]
        # Replicate to 3 channels (matching WMCA's grayscale color format)
        visible = np.stack([ch0, ch0, ch0], axis=-1)

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, visible, [cv2.IMWRITE_JPEG_QUALITY, 95])

    # Step 2: Create 2-column list files from 6-column multi lists
    print('\nCreating single-modal list files...')
    protocols_dir = os.path.join(hqwmca_root, 'protocols')

    for prot_name in sorted(os.listdir(protocols_dir)):
        prot_dir = os.path.join(protocols_dir, prot_name)
        if not os.path.isdir(prot_dir):
            continue

        for split in ['train', 'val', 'test']:
            multi_list = os.path.join(prot_dir, f'{split}_list_multi.txt')
            single_list = os.path.join(prot_dir, f'{split}_list_visible.txt')

            if not os.path.exists(multi_list):
                continue

            entries = []
            with open(multi_list) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 6:
                        continue
                    # Column 1 is color path: frames_v2/.../color_frame_NNNN.jpg
                    color_path = parts[1]
                    label = parts[5]
                    # Convert to visible path
                    visible_path = color_path.replace('frames_v2/', 'frames_visible/').replace('color_', 'visible_')
                    entries.append(f'{visible_path} {label}')

            with open(single_list, 'w') as fh:
                fh.write('\n'.join(entries) + '\n')

            print(f'  {prot_name}/{split}_list_visible.txt: {len(entries)} entries')

    print('\nDone.')


if __name__ == '__main__':
    main()
