#!/usr/bin/env python3
"""
Preprocess the OULU-NPU dataset: extract face-cropped frames from .avi videos
using bbox annotation files, and generate 2-column list files for training.

Input structure:
    datasets/OULU-NPU/{Train,Dev,Test}_files/{video_id}.avi
    datasets/OULU-NPU/{Train,Dev,Test}_files/{video_id}.txt  (bbox: frame_idx,x1,y1,x2,y2)
    datasets/OULU-NPU/Protocols/Protocol_{N}/{Train,Dev,Test}[_{sub}].txt

Output:
    datasets/OULU-NPU-1/frames/{video_id}/frame_{idx:04d}.jpg
    datasets/OULU-NPU-1/Prot/Protocol_{N}/{Split}[_{sub}].txt  (2-column: rel_path label)

Usage:
    python data_process/preprocess_oulu.py [--frame_interval 5] [--oulu_root PATH]
    python data_process/preprocess_oulu.py --debug
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Bbox reading
# ---------------------------------------------------------------------------

def read_bbox_file(bbox_path):
    """
    Read bbox annotation file.
    Format per line: frame_idx,cx,cy,w,h  (center-based)
    Returns dict: {frame_idx: (cx, cy, w, h)}
    """
    bboxes = {}
    if not os.path.exists(bbox_path):
        return bboxes
    with open(bbox_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 5:
                continue
            try:
                frame_idx = int(parts[0])
                cx, cy, bw, bh = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                bboxes[frame_idx] = (cx, cy, bw, bh)
            except ValueError:
                continue
    return bboxes


def crop_face(image, bbox, padding=0.2, output_size=224):
    """
    Crop face from image with padding, resize to output_size x output_size.
    bbox: (cx, cy, w, h) — center-based bounding box.
    """
    h, w = image.shape[:2]
    cx, cy, bw, bh = bbox
    x1 = cx - bw // 2
    y1 = cy - bh // 2
    x2 = cx + bw // 2
    y2 = cy + bh // 2

    pad_w = int(bw * padding)
    pad_h = int(bh * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (output_size, output_size))
    return face


# ---------------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------------

def extract_video_frames(video_path, bbox_path, output_dir, frame_interval, padding=0.2):
    """
    Extract face-cropped frames from one video.
    Returns list of saved filenames (basename only).
    """
    saved = []
    bboxes = read_bbox_file(bbox_path)
    if not bboxes:
        print(f'  [WARN] No bboxes for {video_path}')
        return saved

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  [WARN] Cannot open video: {video_path}')
        return saved

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0 and frame_idx in bboxes:
            face = crop_face(frame, bboxes[frame_idx], padding=padding)
            if face is not None:
                fname = f'frame_{frame_idx:04d}.jpg'
                cv2.imwrite(os.path.join(output_dir, fname), face,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append(fname)

        frame_idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Protocol reading
# ---------------------------------------------------------------------------

def read_protocol_file(prot_path):
    """
    Read OULU-NPU protocol file.
    Format per line: {+1|-1},{video_id}
    Returns list of (video_id, label) where label is 0 or 1.
    """
    entries = []
    if not os.path.exists(prot_path):
        return entries
    with open(prot_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 2:
                continue
            label_str = parts[0].strip()
            video_id = parts[1].strip()
            label = 1 if label_str == '+1' or label_str == '1' else 0
            entries.append((video_id, label))
    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess OULU-NPU: .avi videos -> face-cropped JPEG + list files')
    parser.add_argument('--oulu_root', type=str, default=None,
                        help='Path to OULU-NPU root dir (default: auto-detect)')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Sample every Nth frame (default: 5)')
    parser.add_argument('--padding', type=float, default=0.2,
                        help='Face crop padding ratio (default: 0.2, use 0.0 for tight crops)')
    parser.add_argument('--debug', action='store_true',
                        help='Print info about first video and exit')
    args = parser.parse_args()

    # Resolve root
    if args.oulu_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        oulu_root = os.path.join(project_root, 'datasets', 'OULU-NPU')
    else:
        oulu_root = args.oulu_root

    output_root = os.path.join(os.path.dirname(oulu_root), 'OULU-NPU-1')

    if not os.path.isdir(oulu_root):
        raise FileNotFoundError(f'OULU-NPU root not found: {oulu_root}')

    print(f'OULU-NPU root  : {oulu_root}')
    print(f'Output root    : {output_root}')
    print(f'Frame interval : every {args.frame_interval} frame(s)')
    print(f'Face padding   : {args.padding}')

    # --debug
    if args.debug:
        for split_dir in ['Train_files', 'Dev_files', 'Test_files']:
            d = os.path.join(oulu_root, split_dir)
            if not os.path.isdir(d):
                continue
            avis = sorted(f for f in os.listdir(d) if f.endswith('.avi'))
            if avis:
                vid = avis[0]
                vid_path = os.path.join(d, vid)
                bbox_path = os.path.join(d, vid.replace('.avi', '.txt'))
                cap = cv2.VideoCapture(vid_path)
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                bboxes = read_bbox_file(bbox_path)
                print(f'\n{split_dir}/{vid}: {n_frames} frames, {len(bboxes)} bboxes')
                if bboxes:
                    first_key = sorted(bboxes.keys())[0]
                    print(f'  First bbox: frame {first_key} -> {bboxes[first_key]}')
                break
        return

    # ===== Step 1: Extract frames from all videos =====
    frames_root = os.path.join(output_root, 'frames')
    # Track which video_ids have frames
    video_frames = {}  # video_id -> list of frame filenames

    for split_dir in ['Train_files', 'Dev_files', 'Test_files']:
        split_path = os.path.join(oulu_root, split_dir)
        if not os.path.isdir(split_path):
            print(f'  [WARN] {split_path} not found, skipping')
            continue

        avis = sorted(f for f in os.listdir(split_path) if f.endswith('.avi'))
        print(f'\n{"="*60}')
        print(f'{split_dir}: {len(avis)} videos')
        print('='*60)

        for avi_name in tqdm(avis, desc=split_dir, unit='vid'):
            video_id = avi_name.replace('.avi', '')
            video_path = os.path.join(split_path, avi_name)
            bbox_path = os.path.join(split_path, video_id + '.txt')
            output_dir = os.path.join(frames_root, video_id)

            # Resume: reuse already-extracted frames
            if os.path.isdir(output_dir):
                existing = sorted(f for f in os.listdir(output_dir)
                                  if f.endswith('.jpg'))
                if existing:
                    video_frames[video_id] = existing
                    continue

            saved = extract_video_frames(video_path, bbox_path, output_dir,
                                         args.frame_interval, padding=args.padding)
            video_frames[video_id] = saved

    print(f'\nExtracted frames for {len(video_frames)} videos')

    # ===== Step 2: Generate protocol list files =====
    protocols_dir = os.path.join(oulu_root, 'Protocols')
    if not os.path.isdir(protocols_dir):
        print(f'  [WARN] Protocols dir not found: {protocols_dir}')
        return

    for prot_num in [1, 2, 3, 4]:
        prot_dir = os.path.join(protocols_dir, f'Protocol_{prot_num}')
        if not os.path.isdir(prot_dir):
            print(f'  [SKIP] Protocol_{prot_num} dir not found')
            continue

        out_prot_dir = os.path.join(output_root, 'Prot', f'Protocol_{prot_num}')
        os.makedirs(out_prot_dir, exist_ok=True)

        # Determine split files for this protocol
        if prot_num in [1, 2]:
            split_files = [('Train', 'Train.txt'),
                           ('Dev', 'Dev.txt'),
                           ('Test', 'Test.txt')]
        else:
            split_files = []
            for sub in range(1, 7):
                split_files.append(('Train', f'Train_{sub}.txt'))
                split_files.append(('Dev', f'Dev_{sub}.txt'))
                split_files.append(('Test', f'Test_{sub}.txt'))

        print(f'\n--- Protocol {prot_num} ---')

        for split_name, filename in split_files:
            src_path = os.path.join(prot_dir, filename)
            if not os.path.exists(src_path):
                continue

            prot_entries = read_protocol_file(src_path)
            if not prot_entries:
                continue

            # Generate list entries
            list_entries = []
            for video_id, label in prot_entries:
                frames = video_frames.get(video_id, [])
                for fname in frames:
                    rel_path = f'frames/{video_id}/{fname}'
                    list_entries.append(f'{rel_path} {label}')

            # Write output
            out_path = os.path.join(out_prot_dir, filename)
            with open(out_path, 'w') as fh:
                fh.write('\n'.join(list_entries) + '\n')

            real_n = sum(1 for e in list_entries if e.endswith(' 1'))
            spoof_n = sum(1 for e in list_entries if e.endswith(' 0'))
            print(f'  {filename}: {len(list_entries)} entries '
                  f'(real: {real_n}, spoof: {spoof_n})')

    print('\nPreprocessing complete.')


if __name__ == '__main__':
    main()
