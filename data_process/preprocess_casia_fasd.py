#!/usr/bin/env python3
"""
Preprocess the CASIA-FASD dataset: extract face-cropped frames from .avi videos
using MTCNN face detection, and generate 2-column list files for training.

Input structure:
    datasets/CASIA-FASD/{train,test}_release/{subject_id}/{video}.avi
    Real videos: 1.avi, 2.avi, HR_1.avi (label=1)
    Attack videos: 3-8.avi, HR_2-HR_4.avi (label=0)

Output:
    datasets/CASIA-FASD/frames/{split}/{subject}/{video_stem}/frame_{idx:04d}.jpg
    datasets/CASIA-FASD/protocol/train_list.txt
    datasets/CASIA-FASD/protocol/val_list.txt   (train subjects 16-20 for val)
    datasets/CASIA-FASD/protocol/test_list.txt

Dependency: pip install facenet-pytorch

Usage:
    python data_process/preprocess_casia_fasd.py [--frame_interval 5]
    python data_process/preprocess_casia_fasd.py --debug
"""
import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

REAL_VIDEOS = {'1', '2', 'HR_1'}


def get_label(video_stem):
    """Return 1 for real, 0 for attack based on video filename."""
    return 1 if video_stem in REAL_VIDEOS else 0


def crop_face_bbox(image, bbox, padding=0.2, output_size=224):
    """
    Crop face from image given bbox [x1, y1, x2, y2] with padding.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    bw, bh = x2 - x1, y2 - y1
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


def extract_video_mtcnn(video_path, output_dir, frame_interval, detector):
    """
    Extract face-cropped frames from one video using MTCNN.
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
            # MTCNN expects RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, _ = detector.detect(torch.from_numpy(frame_rgb).unsqueeze(0)
                                        if False else frame_rgb)

            if boxes is not None and len(boxes) > 0:
                # Take the largest face
                areas = [(b[2]-b[0]) * (b[3]-b[1]) for b in boxes]
                best_idx = np.argmax(areas)
                bbox = boxes[best_idx]

                face = crop_face_bbox(frame, bbox)
                if face is not None:
                    fname = f'frame_{frame_idx:04d}.jpg'
                    cv2.imwrite(os.path.join(output_dir, fname), face,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved.append(fname)

        frame_idx += 1

    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess CASIA-FASD: .avi videos -> MTCNN face-cropped '
                    'JPEG + list files')
    parser.add_argument('--casia_root', type=str, default=None,
                        help='Path to CASIA-FASD root dir (default: auto-detect)')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='Sample every Nth frame (default: 5)')
    parser.add_argument('--val_subjects', type=str, default='16,17,18,19,20',
                        help='Comma-separated subject IDs for validation split '
                             '(default: 16,17,18,19,20)')
    parser.add_argument('--debug', action='store_true',
                        help='Print info about first video and exit')
    args = parser.parse_args()

    # Resolve root
    if args.casia_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        casia_root = os.path.join(project_root, 'datasets', 'CASIA-FASD')
    else:
        casia_root = args.casia_root

    if not os.path.isdir(casia_root):
        raise FileNotFoundError(f'CASIA-FASD root not found: {casia_root}')

    val_subjects = set(args.val_subjects.split(','))

    print(f'CASIA-FASD root : {casia_root}')
    print(f'Frame interval  : every {args.frame_interval} frame(s)')
    print(f'Val subjects    : {val_subjects}')

    # --debug
    if args.debug:
        for split in ['train_release', 'test_release']:
            d = os.path.join(casia_root, split)
            if not os.path.isdir(d):
                continue
            subjects = sorted(os.listdir(d))
            if subjects:
                subj_dir = os.path.join(d, subjects[0])
                if os.path.isdir(subj_dir):
                    vids = sorted(f for f in os.listdir(subj_dir)
                                  if f.endswith('.avi'))
                    print(f'\n{split}/{subjects[0]}: {len(vids)} videos')
                    if vids:
                        vid_path = os.path.join(subj_dir, vids[0])
                        cap = cv2.VideoCapture(vid_path)
                        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        cap.release()
                        print(f'  {vids[0]}: {n_frames} frames')
                break
        return

    # Initialize MTCNN
    try:
        from facenet_pytorch import MTCNN
        import torch
    except ImportError:
        raise ImportError('facenet-pytorch is required. Install: pip install facenet-pytorch')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'MTCNN device    : {device}')
    detector = MTCNN(keep_all=True, device=device)

    # Process splits
    frames_root = os.path.join(casia_root, 'frames')
    split_entries = {'train': [], 'val': [], 'test': []}

    split_map = {
        'train_release': 'train',
        'test_release': 'test',
    }

    for src_split, dst_split_base in split_map.items():
        src_dir = os.path.join(casia_root, src_split)
        if not os.path.isdir(src_dir):
            print(f'  [WARN] {src_dir} not found, skipping')
            continue

        subjects = sorted(d for d in os.listdir(src_dir)
                          if os.path.isdir(os.path.join(src_dir, d)))

        print(f'\n{"="*60}')
        print(f'{src_split}: {len(subjects)} subjects')
        print('='*60)

        for subject in tqdm(subjects, desc=src_split, unit='subj'):
            subj_dir = os.path.join(src_dir, subject)
            videos = sorted(f for f in os.listdir(subj_dir) if f.endswith('.avi'))

            # Determine output split
            if dst_split_base == 'train':
                dst_split = 'val' if subject in val_subjects else 'train'
            else:
                dst_split = dst_split_base

            for vid_name in videos:
                video_stem = vid_name.replace('.avi', '')
                label = get_label(video_stem)
                video_path = os.path.join(subj_dir, vid_name)
                output_dir = os.path.join(frames_root, dst_split_base,
                                          subject, video_stem)

                # Resume: reuse already-extracted frames
                if os.path.isdir(output_dir):
                    existing = sorted(f for f in os.listdir(output_dir)
                                      if f.endswith('.jpg'))
                    if existing:
                        for fname in existing:
                            rel = f'frames/{dst_split_base}/{subject}/{video_stem}/{fname}'
                            split_entries[dst_split].append(f'{rel} {label}')
                        continue

                saved = extract_video_mtcnn(video_path, output_dir,
                                            args.frame_interval, detector)
                for fname in saved:
                    rel = f'frames/{dst_split_base}/{subject}/{video_stem}/{fname}'
                    split_entries[dst_split].append(f'{rel} {label}')

    # Write list files
    protocol_dir = os.path.join(casia_root, 'protocol')
    os.makedirs(protocol_dir, exist_ok=True)

    for split_name, entries in split_entries.items():
        out_path = os.path.join(protocol_dir, f'{split_name}_list.txt')
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(entries) + '\n')

        real_n = sum(1 for e in entries if e.endswith(' 1'))
        spoof_n = sum(1 for e in entries if e.endswith(' 0'))
        print(f'\n  {split_name}_list.txt: {len(entries)} entries '
              f'(real: {real_n}, spoof: {spoof_n})')

    print('\nPreprocessing complete.')


if __name__ == '__main__':
    main()
