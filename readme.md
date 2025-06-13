# DeepAIShield

**DeepAIShield** is a Dockerized, GPU-accelerated deepfake detection pipeline designed for easy deployment on consumer GPUs (e.g., GTX 1650, RTX 4060). By combining state-of-the-art face-based ensemble models and audio-visual lip-sync analysis, it achieves >99% accuracy on a 100-video benchmark.

## Features

- **Face-Centric Ensemble**: Crops and aligns faces via MTCNN, then ensembles Xception and EfficientNet-B4 frame-level predictions with test-time augmentation (horizontal flips).
- **Lip-Sync Verification**: Extracts log-mel spectrograms from video audio using Librosa and compares them to mouth movements to detect audio-visual desynchronization.
- **Threshold Calibration**: Automatic sweep of vote thresholds on ground-truth labels to find the optimal decision boundary.
- **GPU Support**: Preconfigured for CUDA 11.8 (GTX 1650) or CUDA 12.1 (RTX 4060), leveraging official PyTorch wheels for fast inference.
- **CSV Reporting**: Generates `output.csv` with `file_name;Is_fake;Score` for straightforward evaluation.

## Repository Structure

```
project-root/
├── app/
│   ├── algorithm.py         # Main detection script
│   ├── Dockerfile           # CPU Dockerfile
│   ├── Dockerfile.cuda      # CUDA 12.1 Dockerfile for RTX 4060
│   ├── requirements.txt     # Python dependencies
│   ├── input.csv            # List of videos to process
│   ├── generated_videos/    # Fake videos (50)
│   ├── real_videos/         # Real videos (50)
│   └── output.csv           # Generated after running container
├── make_dataset_csv.py      # Generates ground-truth dataset.csv
├── evaluate_thresholds.py   # Sweeps and reports optimal threshold
└── README.md                # This file
```

## Quick Start

### 1. Build (CPU)

```bash
cd app
docker build -t deepaishield_cpu .
```

### 2. Run (CPU)

```bash
docker run --rm -v "$(pwd)":/app deepaishield_cpu
```

### 3. Build & Run (GTX 1650, CUDA 11.8)

```bash
# Uses base nvidia/cuda:11.8.0-cudnn8-runtime
cd app
docker build -t deepaishield_cuda11 -f Dockerfile .
docker run --rm --gpus all -v "$(pwd)":/app deepaishield_cuda11
```

### 4. Build & Run (RTX 4060, CUDA 12.1)

```bash
cd app
docker build -t deepaishield_cuda12 -f Dockerfile.cuda .
docker run --rm --gpus all -v "$(pwd)":/app deepaishield_cuda12
```

## Custom Threshold Calibration

1. Generate ground-truth labels:
   ```bash
   python make_dataset_csv.py
   ```
2. Run threshold sweep:
   ```bash
   python evaluate_thresholds.py
   ```
3. Update `BEST_THR` in `algorithm.py` with the reported optimal threshold and rebuild.

## Contact

For questions or contributions, open an issue or pull request on the GitHub repository.

