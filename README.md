# EGA-FAS

Code for `EGA-FAS`, including the ablation-study, benchmarking, and cross-dataset evaluation work used to design, train, and analyze the model.

This repository includes:

- Single-modal and multi-modal training paths
- GE-ShuffleNet backbones and ablation variants
- Adaptive guidance / hard-soft-Gumbel fusion options
- Cross-dataset evaluation setups
- Dataset preprocessing utilities for newer datasets used in the study
- Benchmarking and interpretability scripts

## Paper model

The model used in the paper is:

- `EGA-FAS`
- Backbone: `ShffleNetV2_hd_v1_hybrid_d`
- Fusion setup: Hybrid-D with soft fusion
- Multi-modal settings: `--adaptive_guidance --fusion_type=soft --balance_loss_weight=0.15 --guidance_temperature=0.5`
- Training setup: 40 epochs, cosine repeat LR, SGD, `image_size=64`, `batch_size=64`

## What is in this repo

Main entrypoints:

- `main_FAS.py`: train and evaluate models
- `benchmark_fps.py`: model-only FPS benchmarking
- `benchmark_jetson.py`: Jetson / deployment-oriented benchmarking
- `interpret.py`: guidance-weight and failure analysis

Core folders:

- `config/`: CLI argument parsing and runtime config
- `model/`: model definitions and hybrid backbone variants
- `train_test/`: train / eval loops
- `data_process/`: dataset loading and preprocessing scripts
- `loss/`: loss functions, metrics, optimizer, LR scheduling
- `results/`: training logs, checkpoints, test reports

## Environment

Validated against the local conda environment `LFAS-env`:

```bash
python==3.8.13
torch==2.2.2+cu121
torchvision==0.17.2+cu121
numpy==1.24.4
opencv-python==4.12.0.88
imgaug==0.4.0
timm==0.5.4
yacs==0.1.8
torchsummary==1.5.1
h5py==3.11.0
matplotlib==3.7.5
scikit-image==0.21.0
scikit-learn==1.0.2
scipy==1.10.1
pywavelets==1.4.1
pillow==10.2.0
tqdm==4.64.1
onnx==1.17.0
```

Optional extras:

- `seaborn` for prettier interpretability plots
- `facenet-pytorch` for `CASIA-FASD` / MTCNN-based preprocessing
- `pyrealsense2` for realtime / RealSense-based scripts

Notes:

- `torchaudio` is not required by the current codebase.
- If you install `torchaudio`, it must match the installed `torch` version.

## Dataset layout

The loaders expect datasets under:

```text
datasets/
```

relative to the project root.

Supported dataset keys in the current code include:

- `OULU-NPU`
- `CASIA`
- `RA`
- `MSU`
- `SIW`
- `CASIA-SURF`
- `WMCA`
- `HQ-WMCA`
- `VFPAD`
- `CASIA-FASD`
- `Replay-Attack`
- `CASIA-RA`
- `RA-CASIA`
- `CASIA_FASD-RA`
- `RA-CASIA_FASD`
- `WMCA-HQWMCA`
- `HQWMCA-WMCA`

Several datasets need preprocessing before training. Available utilities:

- `data_process/preprocess_hqwmca.py`
- `data_process/preprocess_casia_fasd.py`
- `data_process/preprocess_replay_attack.py`
- `data_process/preprocess_vfpad.py`
- `data_process/preprocess_oulu.py`
- `data_process/preprocess_oulu_mtcnn.py`
- `data_process/generate_wmca_lists.py`
- `data_process/extract_hqwmca_visible.py`
- `data_process/extract_wmca_images.py`

Examples:

```bash
python data_process/preprocess_hqwmca.py --frame_interval 5
python data_process/preprocess_casia_fasd.py --frame_interval 5
python data_process/preprocess_replay_attack.py --frame_interval 5
python data_process/preprocess_vfpad.py --frame_interval 1
```

## Training

The current CLI uses `--num_modalities` to switch between single-modal and multi-modal runs:

- `--num_modalities 1`: single-modal
- `--num_modalities 3`: color + depth + ir
- `--num_modalities 4`: color + depth + ir + thermal

Common arguments:

- `--model`: model name from `model/bulid_model.py`
- `--dataset_name`: dataset key listed above
- `--prot`: protocol name or ID depending on dataset
- `--sub_prot`: required for some `OULU-NPU` / `SIW` protocols
- `--image_modality`: single-modal input such as `color`, `depth`, `ir`, `thermal`, `rgb`, `ycbcr`
- `--adaptive_guidance`: enable learned per-sample guidance selection
- `--fusion_type soft|hard|gumbel`: override the fusion strategy
- `--strong_augment`: enable stronger augmentation

Examples:

```bash
# Multi-modal WMCA, 3 modalities
python main_FAS.py \
  --model ShffleNetV2_hd_v1_hybrid_d \
  --dataset_name WMCA \
  --prot prints \
  --image_size 64 \
  --batch_size 64 \
  --num_modalities 3

# Multi-modal WMCA with adaptive guidance
python main_FAS.py \
  --model ShffleNetV2_hd_v1_hybrid_d \
  --dataset_name WMCA \
  --prot grandtest \
  --image_size 64 \
  --batch_size 64 \
  --num_modalities 3 \
  --adaptive_guidance

# Paper model: EGA-FAS
python main_FAS.py \
  --model ShffleNetV2_hd_v1_hybrid_d \
  --dataset_name WMCA \
  --prot grandtest \
  --image_size 64 \
  --batch_size 64 \
  --epochs 40 \
  --num_modalities 3 \
  --adaptive_guidance \
  --fusion_type soft \
  --balance_loss_weight 0.15 \
  --guidance_temperature 0.5

# Single-modal HQ-WMCA visible baseline
python main_FAS.py \
  --model ShffleNetV2_hd_v1_hybrid_d \
  --dataset_name HQ-WMCA \
  --prot grand_test-curated \
  --image_modality color \
  --image_size 64 \
  --batch_size 64 \
  --num_modalities 1

# Cross-dataset single-modal evaluation: CASIA-FASD -> Replay-Attack
python main_FAS.py \
  --model ShffleNetV2_hd_v1_hybrid_d \
  --dataset_name CASIA_FASD-RA \
  --image_size 64 \
  --batch_size 64 \
  --num_modalities 1
```

Notes:

- For most training runs, `config/config.py` sets `cycle=10`, so training is repeated multiple times unless the dataset/mode overrides that.
- Multi-modal runs are selected by `--num_modalities`; older examples using `--is_Multi=True` are stale.
- The project uses the code spelling `ShffleNet...`; keep that exact model name in commands.

## Evaluation

Use `--mode infer_test` to evaluate checkpoints saved under the expected results directory:

```bash
python main_FAS.py \
  --model ShffleNetV2_hd_v1_hybrid_d \
  --dataset_name WMCA \
  --prot fakehead \
  --image_size 64 \
  --batch_size 64 \
  --num_modalities 3 \
  --mode infer_test
```

At the moment, `main_FAS.py` resets `config.pretrained_model` before inference, so `infer_test` effectively iterates over all `.pth` files in the run's `checkpoint/` folder rather than honoring a single checkpoint name from the CLI.

## Results layout

Outputs are written under `results/`.

Typical save paths:

```text
results/<dataset>/fusion/<model>_Multi_<image_size>/<prot>/
results/<dataset>/<image_modality>/<model>_Single_<image_size>/
results/<dataset>/<model>_Single_<image_size>/<prot>/<sub_prot>/
```

Each run writes:

- a training log `.txt`
- a `checkpoint/` directory
- `*_test.txt` reports for inference runs

## Useful model names

Examples currently wired in `model/bulid_model.py`:

- `ShffleNetV2_hd_v1`
- `ShffleNetV2_hd_v1_hybrid_a`
- `ShffleNetV2_hd_v1_hybrid_b`
- `ShffleNetV2_hd_v1_hybrid_c`
- `ShffleNetV2_hd_v1_hybrid_d`
- `ShffleNetV2_hd_v1_ablation_eca`
- `ShffleNetV2_hd_v1_ablation_ghost`
- `ShffleNetV2_hd_v1_ablation_adaptive`
- `ECA_FAS_ir`
- `FeatherNetA`
- `FeatherNetB`

## Benchmarking and analysis

Examples:

```bash
python benchmark_fps.py --model ShffleNetV2_hd_v1_hybrid_d
python benchmark_jetson.py --help
python interpret.py --checkpoint path/to/model.pth --dataset_name WMCA --prot prot5
```

## References

The codebase builds on ideas and code from:

- https://github.com/HeDan-11/LFAS-CFMMF
- https://github.com/huawei-noah/Efficient-AI-Backbones
- https://github.com/BangguWu/ECANet

## Citation

Formal citation for `EGA-FAS` is not ready yet and will be added later.

## Contact

Any questions: `mruiz6@uwo.ca`
