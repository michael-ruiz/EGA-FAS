#!/usr/bin/env python3
"""
WMCA Protocol List File Generator

Generates train/dev/test list files for WMCA protocols.
Based on WMCA official protocol definitions and subject-disjoint splits.

Protocols:
- prot1: rigidmask
- prot2: replay
- prot3: prints
- prot4: papermask
- prot5: grandtest (all attacks)
- prot6: glasses
- prot7: flexiblemask
- prot8: fakehead

Author: Claude Code
Date: 2025-12-27
"""

import os
import json
import random
from pathlib import Path
from collections import defaultdict
import argparse


class WMCAListGenerator:
    def __init__(self, image_root, output_root, seed=42):
        """
        Args:
            image_root: Path to preprocessed images directory
            output_root: Path to output PORT2 directory for list files
        """
        self.image_root = Path(image_root)
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

        random.seed(seed)

        # Type ID to protocol mapping (from Bob toolkit)
        self.type_mapping = {
            '0': 'bonafide',
            '1': 'facial_disguise',  # includes glasses
            '2': 'fake_face',  # includes rigid/flexible/paper masks, fakehead
            '3': 'photo',  # prints
            '4': 'video'   # replay
        }

        # For simplicity, map entire type IDs to protocols
        # This is a pragmatic approach given limited PAI mapping info
        self.protocol_type_map = {
            'prot2': ['4'],  # replay = type 4 (video)
            'prot3': ['3'],  # prints = type 3 (photo)
            'prot5': ['1', '2', '3', '4'],  # grandtest = all attacks
            'prot6': ['1'],  # glasses = type 1 (facial disguise)
            # prot1, prot4, prot7, prot8 all come from type 2 (fake_face)
            # Without exact PAI mappings, we'll use type 2 for all of these
            'prot1': ['2'],  # rigidmask
            'prot4': ['2'],  # papermask
            'prot7': ['2'],  # flexiblemask
            'prot8': ['2'],  # fakehead
        }

        self.file_metadata = []

    def scan_images(self):
        """Scan all extracted images and build metadata"""
        print("Scanning extracted images...")

        # Scan color directory (representative of all modalities)
        color_dir = self.image_root / 'color'

        file_dict = defaultdict(list)

        for date_folder in sorted(color_dir.iterdir()):
            if not date_folder.is_dir():
                continue

            for img_file in date_folder.glob('*.jpg'):
                # Parse filename: clientid_sessionid_presenterid_typeid_paiid_frameNN.jpg
                filename = img_file.stem  # Remove .jpg
                parts = filename.rsplit('_frame', 1)

                if len(parts) != 2:
                    continue

                base_name = parts[0]
                frame_num = parts[1]

                # Parse base name
                name_parts = base_name.split('_')
                if len(name_parts) != 5:
                    continue

                client_id, session_id, presenter_id, type_id, pai_id = name_parts

                metadata = {
                    'client_id': client_id,
                    'session_id': session_id,
                    'presenter_id': presenter_id,
                    'type_id': type_id,
                    'pai_id': pai_id,
                    'frame_num': frame_num,
                    'base_name': base_name,
                    'date_folder': date_folder.name,
                    'label': 1 if type_id == '0' else 0  # 1=bonafide, 0=attack
                }

                file_dict[client_id].append(metadata)

        print(f"Found {sum(len(v) for v in file_dict.values())} image samples")
        print(f"Unique subjects: {len(file_dict)}")

        return file_dict

    def create_subject_splits(self, file_dict, train_ratio=0.36, dev_ratio=0.33, test_ratio=0.31):
        """Create subject-disjoint train/dev/test splits with stratification

        Using approximately: 36% train, 33% dev, 31% test based on WMCA docs
        Ensures each split has subjects with attack samples, including
        per-attack-type coverage so every attack type appears in every split.
        """
        # Categorize subjects by whether they have attack samples
        subjects_with_attacks = []
        subjects_bonafide_only = []

        for subject, samples in file_dict.items():
            has_attack = any(s['type_id'] != '0' for s in samples)
            if has_attack:
                subjects_with_attacks.append(subject)
            else:
                subjects_bonafide_only.append(subject)

        print(f"\nSubject categorization:")
        print(f"  Subjects with attacks: {len(subjects_with_attacks)}")
        print(f"  Subjects bonafide only: {len(subjects_bonafide_only)}")

        # Shuffle both groups
        random.shuffle(subjects_with_attacks)
        random.shuffle(subjects_bonafide_only)

        # Split subjects with attacks proportionally
        n_attack = len(subjects_with_attacks)
        n_train_attack = int(n_attack * train_ratio)
        n_dev_attack = int(n_attack * dev_ratio)

        # Split bonafide-only subjects proportionally
        n_bonafide = len(subjects_bonafide_only)
        n_train_bonafide = int(n_bonafide * train_ratio)
        n_dev_bonafide = int(n_bonafide * dev_ratio)

        # Combine splits
        train_subjects = set(
            subjects_with_attacks[:n_train_attack] +
            subjects_bonafide_only[:n_train_bonafide]
        )
        dev_subjects = set(
            subjects_with_attacks[n_train_attack:n_train_attack+n_dev_attack] +
            subjects_bonafide_only[n_train_bonafide:n_train_bonafide+n_dev_bonafide]
        )
        test_subjects = set(
            subjects_with_attacks[n_train_attack+n_dev_attack:] +
            subjects_bonafide_only[n_train_bonafide+n_dev_bonafide:]
        )

        # Per-attack-type coverage verification and repair
        # Build subject → attack types mapping
        subject_attack_types = defaultdict(set)
        for subject, samples in file_dict.items():
            for s in samples:
                if s['type_id'] != '0':
                    subject_attack_types[subject].add(s['type_id'])

        # Collect all attack types used across protocols
        all_attack_types = set()
        for types in self.protocol_type_map.values():
            all_attack_types.update(types)

        splits = {'train': train_subjects, 'dev': dev_subjects, 'test': test_subjects}
        split_names = ['train', 'dev', 'test']

        for attack_type in sorted(all_attack_types):
            # Check coverage per split
            for split_name in split_names:
                has_coverage = any(
                    attack_type in subject_attack_types.get(s, set())
                    for s in splits[split_name]
                )
                if has_coverage:
                    continue

                # Find a donor split that has >1 subject with this attack type
                donor = None
                donor_subject = None
                for other_split in split_names:
                    if other_split == split_name:
                        continue
                    candidates = [
                        s for s in splits[other_split]
                        if attack_type in subject_attack_types.get(s, set())
                    ]
                    if len(candidates) > 1:
                        donor = other_split
                        donor_subject = candidates[0]
                        break

                if donor_subject is None:
                    # All subjects with this type are in one split; take from whichever has them
                    for other_split in split_names:
                        if other_split == split_name:
                            continue
                        candidates = [
                            s for s in splits[other_split]
                            if attack_type in subject_attack_types.get(s, set())
                        ]
                        if candidates:
                            donor = other_split
                            donor_subject = candidates[0]
                            break

                if donor_subject:
                    type_name = self.type_mapping.get(attack_type, attack_type)
                    print(f"  Coverage fix: moving subject {donor_subject} "
                          f"from {donor} to {split_name} for type {attack_type} ({type_name})")
                    splits[donor].discard(donor_subject)
                    splits[split_name].add(donor_subject)

        train_subjects = splits['train']
        dev_subjects = splits['dev']
        test_subjects = splits['test']

        print(f"\nSubject splits:")
        print(f"  Train: {len(train_subjects)} subjects")
        print(f"  Dev:   {len(dev_subjects)} subjects")
        print(f"  Test:  {len(test_subjects)} subjects")

        return train_subjects, dev_subjects, test_subjects

    def generate_list_line(self, metadata):
        """Generate a single line for list file

        Format: rgb_path color_path depth_path ir_path thermal_path label
        """
        date = metadata['date_folder']
        base = metadata['base_name']
        frame = metadata['frame_num']
        filename = f"{base}_frame{frame}.jpg"

        # For now, use color for both RGB and color (as per common practice)
        rgb_path = f"preprocessed/color/{date}/{filename}"
        color_path = f"preprocessed/color/{date}/{filename}"
        depth_path = f"preprocessed/depth/{date}/{filename}"
        ir_path = f"preprocessed/ir/{date}/{filename}"
        thermal_path = f"preprocessed/thermal/{date}/{filename}"
        label = metadata['label']

        return f"{rgb_path} {color_path} {depth_path} {ir_path} {thermal_path} {label}\n"

    def generate_protocol_lists(self, protocol_name, attack_types, file_dict,
                                train_subjects, dev_subjects, test_subjects):
        """Generate train/dev/test lists for a specific protocol"""

        train_lines = []
        dev_lines = []
        test_lines = []

        for client_id, samples in file_dict.items():
            # Determine which split this subject belongs to
            if client_id in train_subjects:
                target_list = train_lines
            elif client_id in dev_subjects:
                target_list = dev_lines
            elif client_id in test_subjects:
                target_list = test_lines
            else:
                continue

            for sample in samples:
                type_id = sample['type_id']

                # Include bonafide samples
                if type_id == '0':
                    target_list.append(self.generate_list_line(sample))
                # Include attack samples if they match this protocol
                elif type_id in attack_types:
                    target_list.append(self.generate_list_line(sample))

        # Write list files
        protocol_files = {
            'train': self.output_root / f"{protocol_name}_train_list.txt",
            'dev': self.output_root / f"{protocol_name}_dev_list.txt",
            'test': self.output_root / f"{protocol_name}_test_list.txt"
        }

        for split_name, lines in [('train', train_lines), ('dev', dev_lines), ('test', test_lines)]:
            with open(protocol_files[split_name], 'w') as f:
                f.writelines(lines)

        print(f"{protocol_name}: train={len(train_lines)}, dev={len(dev_lines)}, test={len(test_lines)}")

        return len(train_lines), len(dev_lines), len(test_lines)

    def generate_all_protocols(self):
        """Generate list files for all 8 protocols"""
        print("\n" + "="*80)
        print("Generating WMCA Protocol List Files")
        print("="*80)

        # Scan images
        file_dict = self.scan_images()

        # Create subject splits
        train_subjects, dev_subjects, test_subjects = self.create_subject_splits(file_dict)

        print("\n" + "-"*80)
        print("Generating protocol list files...")
        print("-"*80)

        stats = {}

        # Generate lists for each protocol
        for prot_num in range(1, 9):
            prot_name = f"prot{prot_num}"
            attack_types = self.protocol_type_map.get(prot_name, [])

            train_count, dev_count, test_count = self.generate_protocol_lists(
                prot_name, attack_types, file_dict,
                train_subjects, dev_subjects, test_subjects
            )

            stats[prot_name] = {
                'train': train_count,
                'dev': dev_count,
                'test': test_count
            }

        # Save statistics
        stats_file = self.output_root / 'list_generation_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        print("\n" + "="*80)
        print("List generation complete!")
        print(f"Statistics saved to: {stats_file}")
        print("="*80)


def main():
    # Get project root dynamically
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    parser = argparse.ArgumentParser(description='Generate WMCA protocol list files')
    parser.add_argument('--image_root', type=str,
                       default=str(project_root / 'datasets' / 'WMCA-1' / 'preprocessed'),
                       help='Path to preprocessed images directory')
    parser.add_argument('--output_root', type=str,
                       default=str(project_root / 'datasets' / 'WMCA-1' / 'PORT2'),
                       help='Path to output PORT2 directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    generator = WMCAListGenerator(
        image_root=args.image_root,
        output_root=args.output_root,
        seed=args.seed
    )

    generator.generate_all_protocols()


if __name__ == '__main__':
    main()
