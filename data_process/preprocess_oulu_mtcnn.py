#!/usr/bin/env python3
"""
Preprocess OULU-NPU using MTCNN face detection instead of provided bounding boxes.
Produces tighter, aligned face crops that remove session-specific background.

Usage:
    python data_process/preprocess_oulu_mtcnn.py [--frame_interval 5]
    python data_process/preprocess_oulu_mtcnn.py --debug  # visualize a few crops
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch


# ---------------------------------------------------------------------------
# MTCNN detector (singleton)
# ---------------------------------------------------------------------------
_detector = None

def get_detector():
    global _detector
    if _detector is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        _detector = MTCNN(
            image_size=224,
            margin=20,           # small margin around detected face
            min_face_size=40,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            keep_all=False,      # only largest face
            device=device,
        )
    return _detector


def read_bbox_file(bbox_path):
    """Read provided bbox as fallback. Format: frame_idx,cx,cy,w,h"""
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


def crop_face_fallback(image, bbox, output_size=224):
    """Fallback crop using provided bbox with tight negative padding."""
    h, w = image.shape[:2]
    cx, cy, bw, bh = bbox
    # Use -0.15 padding for tighter crop
    padding = -0.15
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


def detect_and_crop(image, detector, fallback_bbox=None, output_size=224):
    """
    Detect face with MTCNN. If detection fails, fall back to provided bbox.
    Returns cropped face image (output_size x output_size) or None.
    """
    # MTCNN expects RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect face
    boxes, probs = detector.detect(rgb)

    if boxes is not None and len(boxes) > 0 and probs[0] > 0.9:
        # Use MTCNN detection
        x1, y1, x2, y2 = boxes[0].astype(int)
        # Add small margin (10%)
        bw = x2 - x1
        bh = y2 - y1
        margin_w = int(bw * 0.1)
        margin_h = int(bh * 0.1)
        h, w = image.shape[:2]
        x1 = max(0, x1 - margin_w)
        y1 = max(0, y1 - margin_h)
        x2 = min(w, x2 + margin_w)
        y2 = min(h, y2 + margin_h)
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            if fallback_bbox is not None:
                return crop_face_fallback(image, fallback_bbox, output_size)
            return None
        face = cv2.resize(face, (output_size, output_size))
        return face
    else:
        # Fallback to provided bbox
        if fallback_bbox is not None:
            return crop_face_fallback(image, fallback_bbox, output_size)
        return None


# ---------------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------------

def extract_video_frames(video_path, bbox_path, output_dir, frame_interval, detector):
    """Extract MTCNN-cropped face frames from one video."""
    saved = []
    fallback_bboxes = read_bbox_file(bbox_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'  [WARN] Cannot open video: {video_path}')
        return saved

    os.makedirs(output_dir, exist_ok=True)
    frame_idx = 0
    mtcnn_count = 0
    fallback_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            fallback_bbox = fallback_bboxes.get(frame_idx)
            face = detect_and_crop(frame, detector, fallback_bbox)
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
    """Read OULU-NPU protocol file. Format: {+1|-1},{video_id}"""
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
        description='Preprocess OULU-NPU with MTCNN face detection')
    parser.add_argument('--oulu_root', type=str, default=None)
    parser.add_argument('--frame_interval', type=int, default=5)
    parser.add_argument('--debug', action='store_true',
                        help='Show sample crops and exit')
    args = parser.parse_args()

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
    print(f'Face detection : MTCNN (with fallback to provided bbox)')

    detector = get_detector()

    # --debug: show a few sample crops
    if args.debug:
        for split_dir in ['Train_files', 'Test_files']:
            d = os.path.join(oulu_root, split_dir)
            if not os.path.isdir(d):
                continue
            avis = sorted(f for f in os.listdir(d) if f.endswith('.avi'))[:3]
            for avi in avis:
                vid_path = os.path.join(d, avi)
                bbox_path = os.path.join(d, avi.replace('.avi', '.txt'))
                cap = cv2.VideoCapture(vid_path)
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    continue
                fallback_bboxes = read_bbox_file(bbox_path)
                fallback_bbox = fallback_bboxes.get(0)
                face = detect_and_crop(frame, detector, fallback_bbox)
                if face is not None:
                    out_path = f'/tmp/oulu_mtcnn_debug_{split_dir}_{avi.replace(".avi", "")}.jpg'
                    cv2.imwrite(out_path, face)
                    print(f'  Saved debug crop: {out_path}')
        return

    # ===== Step 1: Extract frames from all videos =====
    frames_root = os.path.join(output_root, 'frames')
    video_frames = {}

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
                                         args.frame_interval, detector)
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

            list_entries = []
            for video_id, label in prot_entries:
                frames = video_frames.get(video_id, [])
                for fname in frames:
                    rel_path = f'frames/{video_id}/{fname}'
                    list_entries.append(f'{rel_path} {label}')

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
