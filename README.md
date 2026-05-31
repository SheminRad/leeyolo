# LEE-YOLO: Edge Object Detection with Label Shift & Covariate Shift Adaptation

This repository contains the codebase for **LEE-YOLO**, a custom YOLOv8-nano architecture designed for edge-computing environments. 

The experiments in this project evaluate how well a model can adapt when trained on ground-level road data from one country (Japan) and deployed in another (United States). Because different countries have different frequencies of road damage (Label Shift), the model's confusion matrix is used to estimate the new environment's distribution and adjust the loss function accordingly.

## 1. Environment Setup

This project uses a modified version of the Ultralytics library to inject label-shift weights directly into the loss function. **Please do not run this using a globally installed pip version of Ultralytics.** The code must be executed directly from this directory so it can access the custom `ultralytics/utils/loss.py` file included in this submission.

**Requirements:**
* Python 3.9 or higher
* An NVIDIA GPU (RTX recommended) with CUDA support for accelerated training.
* Standard scientific libraries: `torch`, `numpy`, `matplotlib`

**Installation:**
Open the terminal in the root directory of the project and install the required dependencies:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy matplotlib pandas pyyaml
pip install -r requirements.txt
```

## 2. Dataset Preparation

This project uses the RDD2022 dataset, specifically focusing on the Japan and United States sub-datasets to test cross-country adaptation.

1. Ensure the dataset is located in the root directory under the folder name `clustered_JapanUS`.
2. To ensure validation integrity and prevent data leakage, run the data splitting script. This script randomly extracts exactly 20% of the training images and labels into a dedicated `val` folder for both countries.

```bash
python split_dataset.py
```

## 3. Reproducing the Results

To guarantee reproducibility, a dynamic environment-variable toggle (`LEE_YOLO_ROBUST_MODE`) is built into the training scripts. This allows the framework to seamlessly switch between standard training and custom weighted training without requiring any manual code changes.

### Phase 1: Baseline Training
To establish the baseline performance, run the baseline script. This automatically disables the custom loss function and trains a standard YOLOv8n model on the Japan dataset. This baseline model is required to understand how the architecture naturally makes mistakes.

```bash
python train_normal.py
```

### Phase 2: Calculating Label Shift Weights
Assuming labeled training data for the US deployment environment is unavailable, the new class frequencies cannot be calculated directly. Instead, the baseline model's **confusion matrix** on the Japan data, alongside its unverified predictions on the US images, is used to mathematically estimate the true distribution of road damage in the US.

Run the weight calculation script to generate the weight multipliers. This script will output a file named `domain_weights.pt` into the root directory.
```bash
python calculate_label_shift.py 
```

### Phase 3: Robust Training (LEE-YOLO)
Once `domain_weights.pt` is generated, start the robust training script. This automatically activates the custom loss wrapper (`SafeBCE`), safely loading the confusion-matrix weights into the PyTorch calculations to prepare the model for the US distribution.

```bash
python train_robust.py
```

## 4. Evaluation and Benchmarking

To generate the metric tables and performance benchmarks presented in the report, dedicated evaluation scripts have been provided for both the baseline and robust models across both datasets. 

These scripts run inference at a standard edge-deployment confidence threshold (25%) and export a formatted PNG table containing the Parameters, GFLOPs, Inference Latency, FPS, mAP scores, Precision, and Recall directly to the project folder.

**To evaluate the Baseline Model:**
```bash
python test_metrics_japan.py
python test_metrics_US.py
```

**To evaluate the Robust Model (LEE-YOLO):**
```bash
python robust_test_metrics_japan.py
python robust_test_metrics_US.py
```

Live training charts and validation visuals can be found in the respective `runs/detect/` subdirectories.
## 5. References

* "Enhancing road maintenance through cyber-physical integration: The LEE-YOLO model for drone-assisted pavement crack detection." *IEEE Transactions on Intelligent Transportation Systems*, vol. 26, no. 9, pp. 14169–14178, 2025. DOI: [10.1109/TITS.2025.3540909](https://doi.org/10.1109/TITS.2025.3540909).
