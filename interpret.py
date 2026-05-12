#!/usr/bin/env python3
"""
EGA-FAS Interpretability Analysis

Analyzes the learned GuidanceSelector behavior in the hybrid_d model,
examining per-sample modality selection, failure patterns, and
per-attack/per-modality performance.

Usage:
    # Run full analysis (inference + plots)
    python interpret.py \
        --checkpoint path/to/model.pth \
        --dataset_name WMCA --prot prot5 \
        --output_dir ./interpret_results

    # Re-analyze from cached results (no GPU needed)
    python interpret.py \
        --from_cache ./interpret_results/raw_data/inference_results.npz \
        --output_dir ./interpret_results
"""

import os
import sys
import csv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from types import SimpleNamespace
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

# ──────────────────────────────────────────────
# Attack type mapping for WMCA
# ──────────────────────────────────────────────
ATTACK_MAP = {
    '0': 'bonafide',
    '1': 'glasses',
    '2': 'masks',
    '3': 'prints',
    '4': 'replay',
}
MODALITY_NAMES = ['depth', 'color', 'ir']


def parse_args():
    parser = argparse.ArgumentParser(description='EGA-FAS Interpretability Analysis')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--dataset_name', type=str, default='WMCA')
    parser.add_argument('--prot', type=str, default='prot5')
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='./interpret_results')
    parser.add_argument('--model', type=str, default='ShffleNetV2_hd_v1_hybrid_d',
                        help='Model name (must match bulid_model.py)')
    parser.add_argument('--ir_prior', type=float, default=0.0,
                        help='IR prior value (for ECA_FAS_ir model)')
    parser.add_argument('--guidance_temperature', type=float, default=1.0,
                        help='Guidance softmax temperature')
    parser.add_argument('--from_cache', type=str, default=None,
                        help='Path to cached inference_results.npz (skip inference)')
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'])
    return parser.parse_args()


def build_mock_config(args):
    """Build a config namespace with fields needed by get_model() and FAS_multi_Dataset."""
    return SimpleNamespace(
        model=args.model,
        is_Multi=True,
        is_Wave=False,
        image_modality=None,
        image_size=args.image_size,
        batch_size=args.batch_size,
        dataset_name=args.dataset_name,
        prot=args.prot,
        sub_prot=None,
        mode='infer_test',
        train_fold_index=-1,
        guidance_modality='depth',
        adaptive_guidance=True,
        fusion_type=None,
        guidance_temperature=args.guidance_temperature,
        ir_prior=args.ir_prior,
        pretrained_model=None,
        epochs=40,
        vis_model=True,
    )


def parse_attack_type(color_path):
    """Parse WMCA attack type from color image path.

    Path format: .../019_06_065_4_13_frame21.jpg
    Fields: client_session_presenter_typeId_paiId_frameN
    type_id: 0=bonafide, 1=glasses, 2=masks, 3=prints, 4=replay
    """
    basename = os.path.basename(color_path)
    stem = basename.rsplit('_frame', 1)[0]  # '019_06_065_4_13'
    parts = stem.split('_')
    if len(parts) >= 4:
        type_id = parts[3]
        return ATTACK_MAP.get(type_id, f'unknown_{type_id}')
    return 'unknown'


# ──────────────────────────────────────────────
# Dataset subclass that also returns color path
# ──────────────────────────────────────────────
def make_interp_dataset_class():
    """Import and subclass FAS_multi_Dataset to also return the color path."""
    from data_process.load_multi_data import FAS_multi_Dataset

    class InterpDataset(FAS_multi_Dataset):
        def __getitem__(self, index):
            images, mask, label = super().__getitem__(index)
            # Get color path from val_list (index 1 for WMCA)
            row = self.val_list[index]
            color_path = row[1] if len(row) > 1 else ''
            return images, mask, label, color_path

    return InterpDataset


def interp_collate_fn(batch):
    """Custom collate: stack tensors, collect strings as list."""
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([torch.tensor(b[1]) if not isinstance(b[1], torch.Tensor) else b[1]
                         for b in batch])
    labels = torch.stack([b[2] for b in batch])
    paths = [b[3] for b in batch]
    return images, masks, labels, paths


# ──────────────────────────────────────────────
# Phase 1: Inference
# ──────────────────────────────────────────────
def run_inference(args, config):
    """Run model inference and collect results."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Build model
    from model.bulid_model import get_model
    net = get_model(config, num_class=2)

    # Load checkpoint
    print(f'Loading checkpoint: {args.checkpoint}')
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # Strip DataParallel 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[new_key] = v
    net.load_state_dict(new_state_dict, strict=False)
    net = net.to(device)
    net.eval()

    # Build dataset
    from data_process.get_path import Get_path
    _, val_path, test_path = Get_path(config.dataset_name, config.prot)
    split_path = test_path if args.split == 'test' else val_path

    InterpDataset = make_interp_dataset_class()
    dataset = InterpDataset(
        data_root=split_path['image_dir'],
        list_path=split_path['prot_list'],
        config=config,
        balance=False,
        isVal=args.split,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, collate_fn=interp_collate_fn, drop_last=False,
    )

    # Inference
    all_probs = []
    all_labels = []
    all_guidance = []
    all_paths = []

    print(f'Running inference on {len(dataset)} samples...')
    for images, masks, labels, paths in tqdm(loader):
        b, n, c, h, w = images.size()
        images = images.view(b * n, c, h, w).to(device)

        with torch.no_grad():
            logit, _, _, _, _, guidance_weights, _ = net(images)

            # Average guidance weights over augmentations
            if guidance_weights is not None:
                gw = guidance_weights.view(b, n, 3).mean(dim=1)  # [b, 3]
                all_guidance.append(gw.cpu().numpy())

            # Average logits over augmentations, then softmax
            logit = logit.view(b, n, -1).mean(dim=1)  # [b, num_class]
            prob = F.softmax(logit, dim=1)

        all_probs.append(prob.cpu().numpy())
        all_labels.append(labels.numpy().reshape(-1))
        all_paths.extend(paths)

    probs = np.concatenate(all_probs, axis=0)        # [N, 2]
    labels = np.concatenate(all_labels, axis=0)       # [N]
    guidance = np.concatenate(all_guidance, axis=0) if all_guidance else None  # [N, 3]
    color_paths = np.array(all_paths)                 # [N]

    # Save to cache
    raw_dir = os.path.join(args.output_dir, 'raw_data')
    os.makedirs(raw_dir, exist_ok=True)
    save_path = os.path.join(raw_dir, 'inference_results.npz')
    np.savez(save_path, probs=probs, labels=labels, guidance=guidance, paths=color_paths)
    print(f'Saved raw results to {save_path}')

    return probs, labels, guidance, color_paths


def load_cache(cache_path):
    """Load cached inference results."""
    print(f'Loading cached results from {cache_path}')
    data = np.load(cache_path, allow_pickle=True)
    return data['probs'], data['labels'], data['guidance'], data['paths']


# ──────────────────────────────────────────────
# Phase 2: Guidance Selector Distribution
# ──────────────────────────────────────────────
def analysis_guidance(probs, labels, guidance, paths, output_dir):
    """Analyze guidance selector distribution."""
    out_dir = os.path.join(output_dir, 'analysis1_guidance')
    os.makedirs(out_dir, exist_ok=True)

    if guidance is None:
        print('No guidance weights found (model may not use adaptive guidance). Skipping.')
        return

    N = len(labels)
    attack_types = np.array([parse_attack_type(p) for p in paths])
    selected = np.argmax(guidance, axis=1)  # [N] hard argmax selection

    # Entropy of softmax weights
    eps = 1e-8
    entropy = -np.sum(guidance * np.log(guidance + eps), axis=1)
    max_entropy = np.log(3)

    # ── 1. Overall selection bar chart ──
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = [np.sum(selected == i) for i in range(3)]
    pcts = [c / N * 100 for c in counts]
    bars = ax.bar(MODALITY_NAMES, counts, color=['#2196F3', '#4CAF50', '#FF9800'])
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + N * 0.01,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('Count')
    ax.set_title('Guidance Modality Selection (Hard Argmax)')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'guidance_overall_bar.png'), dpi=150)
    plt.close(fig)

    # ── 2. Selection by attack type ──
    unique_attacks = sorted(set(attack_types))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(unique_attacks))
    width = 0.25
    for mod_idx, (name, color) in enumerate(
            zip(MODALITY_NAMES, ['#2196F3', '#4CAF50', '#FF9800'])):
        vals = []
        for atk in unique_attacks:
            mask = attack_types == atk
            atk_selected = selected[mask]
            vals.append(np.sum(atk_selected == mod_idx) / max(mask.sum(), 1) * 100)
        ax.bar(x + mod_idx * width, vals, width, label=name, color=color)
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Selection %')
    ax.set_title('Modality Selection by Attack Type')
    ax.set_xticks(x + width)
    ax.set_xticklabels(unique_attacks, rotation=15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'guidance_by_attack.png'), dpi=150)
    plt.close(fig)

    # ── 3. Selection by class (real vs spoof) ──
    fig, ax = plt.subplots(figsize=(6, 4))
    class_names = ['spoof (0)', 'bonafide (1)']
    x_cls = np.arange(2)
    for mod_idx, (name, color) in enumerate(
            zip(MODALITY_NAMES, ['#2196F3', '#4CAF50', '#FF9800'])):
        vals = []
        for cls in [0, 1]:
            mask = labels == cls
            cls_selected = selected[mask]
            vals.append(np.sum(cls_selected == mod_idx) / max(mask.sum(), 1) * 100)
        ax.bar(x_cls + mod_idx * width, vals, width, label=name, color=color)
    ax.set_xlabel('Class')
    ax.set_ylabel('Selection %')
    ax.set_title('Modality Selection by Class')
    ax.set_xticks(x_cls + width)
    ax.set_xticklabels(class_names)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'guidance_by_class.png'), dpi=150)
    plt.close(fig)

    # ── 4. Entropy histogram ──
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(entropy, bins=50, color='#9C27B0', alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(max_entropy, color='red', linestyle='--',
               label=f'Max entropy ({max_entropy:.2f})')
    ax.axvline(np.mean(entropy), color='blue', linestyle='--',
               label=f'Mean ({np.mean(entropy):.3f})')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Count')
    ax.set_title('Guidance Weight Entropy Distribution')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'guidance_entropy_hist.png'), dpi=150)
    plt.close(fig)

    # ── 5. Box plots of raw weights ──
    fig, ax = plt.subplots(figsize=(6, 4))
    bp = ax.boxplot([guidance[:, i] for i in range(3)], tick_labels=MODALITY_NAMES,
                    patch_artist=True)
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    ax.set_ylabel('Softmax Weight')
    ax.set_title('Guidance Weight Distribution per Modality')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'guidance_weights_box.png'), dpi=150)
    plt.close(fig)

    # ── 6. Summary CSV ──
    csv_path = os.path.join(out_dir, 'guidance_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'depth', 'color', 'ir'])
        writer.writerow(['selection_count'] +
                        [str(np.sum(selected == i)) for i in range(3)])
        writer.writerow(['selection_pct'] +
                        [f'{np.sum(selected == i) / N * 100:.2f}' for i in range(3)])
        writer.writerow(['mean_weight'] +
                        [f'{guidance[:, i].mean():.4f}' for i in range(3)])
        writer.writerow(['std_weight'] +
                        [f'{guidance[:, i].std():.4f}' for i in range(3)])
        writer.writerow(['min_weight'] +
                        [f'{guidance[:, i].min():.4f}' for i in range(3)])
        writer.writerow(['max_weight'] +
                        [f'{guidance[:, i].max():.4f}' for i in range(3)])
        writer.writerow(['mean_entropy', f'{np.mean(entropy):.4f}', '', ''])
        writer.writerow(['median_entropy', f'{np.median(entropy):.4f}', '', ''])
        # Per-attack breakdown
        writer.writerow([])
        writer.writerow(['attack_type', 'depth_pct', 'color_pct', 'ir_pct', 'count'])
        for atk in unique_attacks:
            mask = attack_types == atk
            atk_n = mask.sum()
            atk_sel = selected[mask]
            writer.writerow([atk] +
                            [f'{np.sum(atk_sel == i) / atk_n * 100:.1f}' for i in range(3)] +
                            [str(atk_n)])

    print(f'Guidance analysis saved to {out_dir}/')


# ──────────────────────────────────────────────
# Phase 3: Failure Cases
# ──────────────────────────────────────────────
def analysis_failures(probs, labels, guidance, paths, output_dir):
    """Analyze failure cases."""
    out_dir = os.path.join(output_dir, 'analysis2_failures')
    os.makedirs(out_dir, exist_ok=True)

    N = len(labels)
    attack_types = np.array([parse_attack_type(p) for p in paths])

    # Predictions: class 1 prob > 0.5 -> bonafide
    bonafide_prob = probs[:, 1]
    pred_labels = (bonafide_prob > 0.5).astype(int)
    correct_mask = pred_labels == labels
    incorrect_mask = ~correct_mask

    # FP: predicted bonafide (1) but actually spoof (0) — missed attack
    fp_mask = (pred_labels == 1) & (labels == 0)
    # FN: predicted spoof (0) but actually bonafide (1) — false alarm
    fn_mask = (pred_labels == 0) & (labels == 1)

    n_correct = correct_mask.sum()
    n_incorrect = incorrect_mask.sum()
    n_fp = fp_mask.sum()
    n_fn = fn_mask.sum()

    unique_attacks = sorted(set(attack_types))

    # Per-attack failure stats (computed once, reused below)
    atk_names = []
    atk_rates = []
    atk_counts = []
    for atk in unique_attacks:
        mask = attack_types == atk
        n_atk = mask.sum()
        n_wrong = (incorrect_mask & mask).sum()
        atk_names.append(atk)
        atk_rates.append(n_wrong / max(n_atk, 1) * 100)
        atk_counts.append(n_atk)

    # ── 1. Failure rate by attack type ──
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(atk_names, atk_rates, color='#E53935', alpha=0.8)
    for bar, cnt in zip(bars, atk_counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f'n={cnt}', ha='center', va='bottom', fontsize=9)
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Misclassification Rate (%)')
    ax.set_title('Failure Rate by Attack Type')
    ax.set_xticks(range(len(atk_names)))
    ax.set_xticklabels(atk_names, rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'failure_rate_by_attack.png'), dpi=150)
    plt.close(fig)

    # ── 2. Guidance weights: correct vs incorrect ──
    if guidance is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for i, (name, ax) in enumerate(zip(MODALITY_NAMES, axes)):
            data_lists = [guidance[correct_mask, i]]
            box_labels = ['correct']
            if n_incorrect > 0:
                data_lists.append(guidance[incorrect_mask, i])
                box_labels.append('incorrect')
            bp = ax.boxplot(data_lists, tick_labels=box_labels, patch_artist=True)
            colors_bp = ['#4CAF50', '#E53935']
            for patch, c in zip(bp['boxes'], colors_bp[:len(data_lists)]):
                patch.set_facecolor(c)
                patch.set_alpha(0.6)
            ax.set_title(f'{name} weight')
            ax.set_ylabel('Weight')
        fig.suptitle('Guidance Weights: Correct vs Incorrect Predictions')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'guidance_correct_vs_incorrect.png'), dpi=150)
        plt.close(fig)

    # ── 3. Score distribution by attack type ──
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_atk = [bonafide_prob[attack_types == atk] for atk in unique_attacks]
    bp = ax.boxplot(data_by_atk, tick_labels=unique_attacks, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#7E57C2')
        patch.set_alpha(0.6)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='threshold=0.5')
    ax.set_xlabel('Attack Type')
    ax.set_ylabel('Bonafide Probability')
    ax.set_title('Prediction Score Distribution by Attack Type')
    ax.legend()
    ax.set_xticks(range(1, len(unique_attacks) + 1))
    ax.set_xticklabels(unique_attacks, rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'score_distribution_by_attack.png'), dpi=150)
    plt.close(fig)

    # ── 4. Misclassified samples CSV ──
    csv_path = os.path.join(out_dir, 'misclassified_samples.csv')
    selected = np.argmax(guidance, axis=1) if guidance is not None else np.full(N, -1)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path', 'true_label', 'pred_label', 'bonafide_prob',
                         'attack_type', 'selected_modality',
                         'depth_weight', 'color_weight', 'ir_weight'])
        for idx in np.where(incorrect_mask)[0]:
            sel_name = MODALITY_NAMES[selected[idx]] if selected[idx] >= 0 else 'N/A'
            gw = guidance[idx] if guidance is not None else [0, 0, 0]
            writer.writerow([
                paths[idx],
                int(labels[idx]),
                int(pred_labels[idx]),
                f'{bonafide_prob[idx]:.4f}',
                attack_types[idx],
                sel_name,
                f'{gw[0]:.4f}', f'{gw[1]:.4f}', f'{gw[2]:.4f}',
            ])

    # ── 5. Failure summary text ──
    summary_path = os.path.join(out_dir, 'failure_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('EGA-FAS Failure Analysis Summary\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Total samples: {N}\n')
        f.write(f'Correct: {n_correct} ({n_correct / N * 100:.2f}%)\n')
        f.write(f'Incorrect: {n_incorrect} ({n_incorrect / N * 100:.2f}%)\n')
        f.write(f'  False Positives (spoof predicted as bonafide): {n_fp}\n')
        f.write(f'  False Negatives (bonafide predicted as spoof): {n_fn}\n\n')
        f.write('Per-attack failure rates:\n')
        f.write('-' * 40 + '\n')
        for atk, rate, cnt in sorted(zip(atk_names, atk_rates, atk_counts),
                                      key=lambda x: -x[1]):
            n_wrong = int(round(rate * cnt / 100))
            f.write(f'  {atk:15s}: {rate:6.2f}% ({n_wrong}/{cnt})\n')

    print(f'Failure analysis saved to {out_dir}/')


# ──────────────────────────────────────────────
# Phase 4: Per-Modality & Per-Attack Performance
# ──────────────────────────────────────────────
def analysis_modality(probs, labels, guidance, paths, output_dir):
    """Per-modality and per-attack performance analysis."""
    out_dir = os.path.join(output_dir, 'analysis3_modality')
    os.makedirs(out_dir, exist_ok=True)

    N = len(labels)
    attack_types = np.array([parse_attack_type(p) for p in paths])
    unique_attacks = sorted(set(attack_types))
    bonafide_prob = probs[:, 1]
    pred_labels = (bonafide_prob > 0.5).astype(int)

    # ── 1. Per-attack metrics ──
    from loss.metric import model_performances

    atk_acers = []
    atk_accs = []
    for atk in unique_attacks:
        mask = attack_types == atk
        if mask.sum() < 2 or len(set(labels[mask])) < 2:
            # Can't compute ACER with only one class
            atk_acers.append(0)
            atk_accs.append(np.mean(pred_labels[mask] == labels[mask]) * 100)
            continue
        try:
            eval_dict, _, _ = model_performances(bonafide_prob[mask], labels[mask])
            atk_acers.append(eval_dict['ACER'] * 100)
            atk_accs.append(eval_dict['ACC'] * 100)
        except Exception:
            atk_acers.append(0)
            atk_accs.append(np.mean(pred_labels[mask] == labels[mask]) * 100)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(unique_attacks, atk_accs, color='#2196F3', alpha=0.8)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy per Attack Type')
    axes[0].set_xticks(range(len(unique_attacks)))
    axes[0].set_xticklabels(unique_attacks, rotation=15)

    axes[1].bar(unique_attacks, atk_acers, color='#E53935', alpha=0.8)
    axes[1].set_ylabel('ACER (%)')
    axes[1].set_title('ACER per Attack Type')
    axes[1].set_xticks(range(len(unique_attacks)))
    axes[1].set_xticklabels(unique_attacks, rotation=15)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'per_attack_metrics.png'), dpi=150)
    plt.close(fig)

    # ── 2. Guidance selection vs accuracy ──
    if guidance is not None:
        selected = np.argmax(guidance, axis=1)

        fig, ax = plt.subplots(figsize=(6, 4))
        mod_accs = []
        mod_counts = []
        for i in range(3):
            mask = selected == i
            if mask.sum() > 0:
                acc = np.mean(pred_labels[mask] == labels[mask]) * 100
            else:
                acc = 0
            mod_accs.append(acc)
            mod_counts.append(mask.sum())
        bars = ax.bar(MODALITY_NAMES, mod_accs,
                      color=['#2196F3', '#4CAF50', '#FF9800'])
        for bar, cnt in zip(bars, mod_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'n={cnt}', ha='center', va='bottom', fontsize=9)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy by Selected Guidance Modality')
        ax.set_ylim(0, 105)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'guidance_vs_accuracy.png'), dpi=150)
        plt.close(fig)

        # ── 3. Heatmap: attack type x modality selection ──
        heatmap_data = np.zeros((len(unique_attacks), 3))
        for ai, atk in enumerate(unique_attacks):
            mask = attack_types == atk
            atk_sel = selected[mask]
            atk_n = mask.sum()
            for mi in range(3):
                heatmap_data[ai, mi] = np.sum(atk_sel == mi) / max(atk_n, 1) * 100

        fig, ax = plt.subplots(figsize=(6, max(4, len(unique_attacks) * 0.8)))
        if HAS_SEABORN:
            sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd',
                        xticklabels=MODALITY_NAMES, yticklabels=unique_attacks, ax=ax)
        else:
            im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
            ax.set_xticks(range(3))
            ax.set_xticklabels(MODALITY_NAMES)
            ax.set_yticks(range(len(unique_attacks)))
            ax.set_yticklabels(unique_attacks)
            for ai in range(len(unique_attacks)):
                for mi in range(3):
                    ax.text(mi, ai, f'{heatmap_data[ai, mi]:.1f}',
                            ha='center', va='center')
            plt.colorbar(im, ax=ax)
        ax.set_title('Modality Selection % by Attack Type')
        ax.set_xlabel('Selected Modality')
        ax.set_ylabel('Attack Type')
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, 'guidance_by_attack_heatmap.png'), dpi=150)
        plt.close(fig)

    # ── 4. Confusion matrix ──
    tp = int(np.sum((pred_labels == 1) & (labels == 1)))
    fp = int(np.sum((pred_labels == 1) & (labels == 0)))
    fn = int(np.sum((pred_labels == 0) & (labels == 1)))
    tn = int(np.sum((pred_labels == 0) & (labels == 0)))
    cm = np.array([[tn, fp], [fn, tp]])

    fig, ax = plt.subplots(figsize=(5, 4))
    if HAS_SEABORN:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['spoof', 'bonafide'],
                    yticklabels=['spoof', 'bonafide'], ax=ax)
    else:
        ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['spoof', 'bonafide'])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['spoof', 'bonafide'])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'confusion_matrix.png'), dpi=150)
    plt.close(fig)

    # ── 5. Per-attack summary CSV ──
    csv_path = os.path.join(out_dir, 'per_attack_summary.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['attack_type', 'count', 'accuracy', 'ACER',
                         'n_correct', 'n_incorrect'])
        for atk, acc, acer in zip(unique_attacks, atk_accs, atk_acers):
            mask = attack_types == atk
            n_atk = mask.sum()
            n_cor = int((pred_labels[mask] == labels[mask]).sum())
            writer.writerow([atk, n_atk, f'{acc:.2f}', f'{acer:.2f}',
                             n_cor, n_atk - n_cor])

    # Overall metrics
    try:
        overall_eval, overall_tpr_fpr, _ = model_performances(bonafide_prob, labels)
        print(f'\nOverall: ACC={overall_eval["ACC"] * 100:.2f}%, '
              f'ACER={overall_eval["ACER"] * 100:.2f}%, '
              f'APCER={overall_eval["APCER"] * 100:.2f}%, '
              f'BPCER={overall_eval["BPCER"] * 100:.2f}%')
    except Exception as e:
        print(f'Could not compute overall metrics: {e}')

    print(f'Per-modality analysis saved to {out_dir}/')


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()

    if args.from_cache:
        probs, labels, guidance, paths = load_cache(args.from_cache)
    else:
        if not args.checkpoint:
            print('ERROR: --checkpoint is required when not using --from_cache')
            sys.exit(1)
        config = build_mock_config(args)
        probs, labels, guidance, paths = run_inference(args, config)

    print(f'\nDataset: {len(labels)} samples, '
          f'{np.sum(labels == 1)} bonafide, {np.sum(labels == 0)} spoof')

    analysis_guidance(probs, labels, guidance, paths, args.output_dir)
    analysis_failures(probs, labels, guidance, paths, args.output_dir)
    analysis_modality(probs, labels, guidance, paths, args.output_dir)

    print(f'\nAll results saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
