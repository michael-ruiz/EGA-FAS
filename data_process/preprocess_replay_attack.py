#!/usr/bin/env python3
"""
Preprocess the Replay-Attack dataset: extract face-cropped frames from .mov
videos using .face annotation files, and generate 2-column list files.

Input structure:
    datasets/Replay-Attack/Replay-Attack/{train,devel,test}/{real,attack/...}/{video}.mov
    datasets/Replay-Attack/Replay-Attack/face-locations/{split}/{...}/{video}.face
    .face format per line: frame_idx x y w h

Output:
    datasets/Replay-Attack/frames/{split}/{rel_video_path}/frame_{idx:04d}.jpg
    datasets/Replay-Attack/protocol/train_list.txt
    datasets/Replay-Attack/protocol/val_list.txt   (from devel split)
    datasets/Replay-Attack/protocol/test_list.txt

Usage:
    python data_process/preprocess_replay_attack.py [--frame_interval 5]
    python data_process/preprocess_replay_attack.py --debug
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Face file reading
# ---------------------------------------------------------------------------

def read_face_file(face_path):
    """
    Read .face annotation file.
    Format per line: frame_idx x y w h
    Returns dict: {frame_idx: (x, y, w, h)}
    """
    bboxes = {}
    if not os.path.exists(face_path):
        return bboxes
    with open(face_path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                frame_idx = int(parts[0])
                x, y, w, h = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                bboxes[frame_idx] = (x, y, w, h)
            except ValueError:
                continue
    return bboxes


def crop_face(image, bbox_xywh, padding=0.2, output_size=224):
    """
    Crop face from image given bbox (x, y, w, h) with padding.
    """
    img_h, img_w = image.shape[:2]
    x, y, w, h = bbox_xywh
    x1, y1, x2, y2 = x, y, x + w, y + h

    pad_w = int(w * padding)
    pad_h = int(h * padding)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img_w, x2 + pad_w)
    y2 = min(img_h, y2 + pad_h)

    face = image[y1:y2, x1:x2]
    if face.size == 0:
        return None
    face = cv2.resize(face, (output_size, output_size))
    return face


# ---------------------------------------------------------------------------
# Video extraction
# ---------------------------------------------------------------------------

def extract_video_frames(video_path, face_path, output_dir, frame_interval):
    """
    Extract face-cropped frames from one video using .face annotations.
    Returns list of saved filenames.
    """
    saved = []
    bboxes = read_face_file(face_path)
    if not bboxes:
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
            face = crop_face(frame, bboxes[frame_idx])
            if face is not None:
                fname = f'frame_{frame_idx:04d}.jpg'
                cv2.imwrite(os.path.join(output_dir, fname), face,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                saved.append(fname)

        frame_idx += 1

    cap.release()
    return saved


def extract_video_mtcnn(video_path, output_dir, frame_interval, detector):
    """
    Extract face-cropped frames from one video using MTCNN (same as CASIA-FASD).
    Returns list of saved filenames.
    """
    import torch

    saved = []
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

        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(frame_rgb)

            if boxes is not None and len(boxes) > 0:
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
                best_idx = np.argmax(areas)
                bbox = boxes[best_idx]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                face = crop_face(frame, (x1, y1, x2 - x1, y2 - y1))
                if face is not None:
                    fname = f'frame_{frame_idx:04d}.jpg'
                    cv2.imwrite(os.path.join(output_dir, fname), face,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved.append(fname)

        frame_idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_label_from_path(rel_path):
    """Determine label from video path: 'real' in path -> 1, 'attack' -> 0."""
    parts = rel_path.replace('\\', '/').split('/')
    for p in parts:
        if p == 'real':
            return 1
        if p == 'attack':
            return 0
    return 0


def find_face_file(ra_inner_root, split, rel_video_path):
    """
    Find the .face file for a given video.
    rel_video_path: e.g. 'real/client001_session01_webcam_authenticate_adverse_1'
    """
    face_dir = os.path.join(ra_inner_root, 'face-locations', split)
    face_path = os.path.join(face_dir, rel_video_path + '.face')
    if os.path.exists(face_path):
        return face_path
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Preprocess Replay-Attack: .mov videos -> face-cropped '
                    'JPEG + list files')
    parser.add_argument('--ra_root', type=str, default=None,
                        help='Path to Replay-Attack root dir (default: auto-detect)')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Sample every Nth frame (default: 5)')
    parser.add_argument('--mtcnn', action='store_true',
                        help='Use MTCNN face detection instead of .face annotations')
    parser.add_argument('--frames_dir', type=str, default='frames',
                        help='Output frames directory name (default: frames)')
    parser.add_argument('--debug', action='store_true',
                        help='Print info about first video and exit')
    args = parser.parse_args()

    # Resolve root
    if args.ra_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        ra_root = os.path.join(project_root, 'datasets', 'Replay-Attack')
    else:
        ra_root = args.ra_root

    if not os.path.isdir(ra_root):
        raise FileNotFoundError(f'Replay-Attack root not found: {ra_root}')

    # The actual data may be inside a Replay-Attack subdirectory
    ra_inner = os.path.join(ra_root, 'Replay-Attack')
    if os.path.isdir(ra_inner):
        ra_inner_root = ra_inner
    else:
        ra_inner_root = ra_root

    print(f'Replay-Attack root : {ra_root}')
    print(f'Inner root         : {ra_inner_root}')
    print(f'Frame interval     : every {args.frame_interval} frame(s)')

    # --debug
    if args.debug:
        for split in ['train', 'devel', 'test']:
            d = os.path.join(ra_inner_root, split)
            if not os.path.isdir(d):
                continue
            # Find first .mov recursively
            for root, dirs, files in os.walk(d):
                movs = [f for f in files if f.endswith('.mov')]
                if movs:
                    vid_path = os.path.join(root, movs[0])
                    cap = cv2.VideoCapture(vid_path)
                    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    rel = os.path.relpath(root, d)
                    vid_stem = movs[0].replace('.mov', '')
                    face_path = find_face_file(ra_inner_root, split,
                                               os.path.join(rel, vid_stem))
                    n_bboxes = len(read_face_file(face_path)) if face_path else 0
                    print(f'\n{split}/{rel}/{movs[0]}: {n_frames} frames, '
                          f'{n_bboxes} face annotations')
                    if face_path:
                        print(f'  Face file: {face_path}')
                    break
            break
        return

    # Initialize MTCNN if requested
    detector = None
    if args.mtcnn:
        try:
            from facenet_pytorch import MTCNN
            import torch
        except ImportError:
            raise ImportError('facenet-pytorch is required for --mtcnn. Install: pip install facenet-pytorch')
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'MTCNN device       : {device}')
        detector = MTCNN(keep_all=True, device=device)

    # Process splits
    # devel -> val in output naming
    split_names = [('train', 'train'), ('devel', 'val'), ('test', 'test')]
    frames_dir = args.frames_dir
    frames_root = os.path.join(ra_root, frames_dir)
    protocol_dir = os.path.join(ra_root, 'protocol')
    os.makedirs(protocol_dir, exist_ok=True)

    for src_split, dst_split in split_names:
        split_dir = os.path.join(ra_inner_root, src_split)
        if not os.path.isdir(split_dir):
            print(f'  [WARN] {split_dir} not found, skipping')
            continue

        print(f'\n{"="*60}')
        print(f'Split: {src_split} -> {dst_split}')
        print('='*60)

        # Find all .mov files recursively
        video_files = []
        for root, dirs, files in os.walk(split_dir):
            for f in files:
                if f.endswith('.mov'):
                    full_path = os.path.join(root, f)
                    rel_dir = os.path.relpath(root, split_dir)
                    video_files.append((full_path, rel_dir, f))

        video_files.sort(key=lambda x: x[0])
        print(f'  Found {len(video_files)} videos')

        entries = []

        for vid_path, rel_dir, vid_name in tqdm(video_files, desc=src_split,
                                                  unit='vid'):
            video_stem = vid_name.replace('.mov', '')
            rel_video_path = os.path.join(rel_dir, video_stem)
            label = get_label_from_path(rel_dir)

            output_dir = os.path.join(frames_root, src_split, rel_dir, video_stem)

            # Resume: reuse already-extracted frames
            if os.path.isdir(output_dir):
                existing = sorted(f for f in os.listdir(output_dir)
                                  if f.endswith('.jpg'))
                if existing:
                    for fname in existing:
                        rel = f'{frames_dir}/{src_split}/{rel_dir}/{video_stem}/{fname}'
                        entries.append(f'{rel} {label}')
                    continue

            if args.mtcnn:
                saved = extract_video_mtcnn(vid_path, output_dir,
                                            args.frame_interval, detector)
            else:
                face_path = find_face_file(ra_inner_root, src_split, rel_video_path)
                if face_path is None:
                    print(f'  [WARN] No face file for {rel_video_path}')
                    continue
                saved = extract_video_frames(vid_path, face_path, output_dir,
                                             args.frame_interval)

            for fname in saved:
                rel = f'{frames_dir}/{src_split}/{rel_dir}/{video_stem}/{fname}'
                entries.append(f'{rel} {label}')

        # Write list file
        suffix = '_mtcnn' if args.mtcnn else ''
        out_path = os.path.join(protocol_dir, f'{dst_split}_list{suffix}.txt')
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(entries) + '\n')

        real_n = sum(1 for e in entries if e.endswith(' 1'))
        spoof_n = sum(1 for e in entries if e.endswith(' 0'))
        print(f'\n  {dst_split}_list.txt: {len(entries)} entries '
              f'(real: {real_n}, spoof: {spoof_n})')

    print('\nPreprocessing complete.')


if __name__ == '__main__':
    main()
