# AthleticsPose: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability (MMSports at ACMMM 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2507.12905-b31b1b.svg)](https://arxiv.org/abs/2507.12905)

![Dataset Example](docs/assets/all_pose_animation.gif)

## Abstract

This repository contains the implementation for our paper "AthleticsPose: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability". We introduce a comprehensive dataset and evaluation framework for 3D pose estimation in athletics scenarios.

## Updates

- **2025-08-10**: Added simple usage of codebase.
- **2025-08-04**: Added dataset download instructions and license files
- **2025-07-18**: Initial repository setup and basic file upload

## Setup

### Clone the Repository
To get started, clone this repository:
```bash
git clone https://github.com/your-username/AthleticsPose.git
cd AthleticsPose
```

### Install Dependencies

Use the provided `Makefile` to set up the environment and download the dataset and pretrained checkpoints.
This requires `make` and [`uv`](https://docs.astral.sh/uv/) to be installed on your system.
We also assume **CUDA 11.8** is available for GPU acceleration.

If you don't have `make`, install it via your package manager:
- Ubuntu/Debian: `sudo apt-get install make`
- macOS: `brew install make`

If you don't have `uv`, install it with:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
or on macOS with Homebrew:
```bash
brew install uv
```

Then run the initial setup:
```bash
make setup
```
This will:
1. Create and install the Python virtual environment with all dependencies
2. Download and extract the dataset and pretrained checkpoints

### Optional commands
Environment only (no dataset/checkpoint download):
```bash
make venv
```

Dataset and checkpoints only (no environment setup):
```bash
make download
```

You can download only the AthleticsPose dataset using the following command:
```bash
curl -L -o data.zip \
  "https://github.com/SZucchini/AthleticsPose/releases/latest/download/data.zip"
```
or
```bash
wget -O data.zip \
  "https://github.com/SZucchini/AthleticsPose/releases/latest/download/data.zip"
```

## Usage

### Evaluate using checkpoints
For the model trained on AthleticsPose dataset, run:
```bash
uv run python scripts/evaluate.py evaluation=default
```

For the model trained on Human3.6M dataset, run:
```bash
uv run python scripts/evaluate.py evaluation=h36m_pretrained
```

For the model trained on AthletePose3D dataset, run:
```bash
uv run python scripts/evaluate.py evaluation=ap3d_pretrained
```

### Train a model from scratch
To train a model from scratch, use the following command:
```bash
uv run python scripts/train.py exp_name=<your_experiment_name> \
  data.input_2d_type=det \  # or gt
  data.det_model_type=ft \  # or pretrained
  wandb.project=<your_wandb_project_name>  # You can use WandB for logging
```

### Predictions and analyses similar to those in the paper
For example, to run the predictions and analyses similar to those in the paper, use:
```bash
uv run python scripts/predict.py prediction=from_2d_markers prediction.input.marker_type=det_ft
uv run python scripts/analyze_predictions.py data/AthleticsPoseDataset/predictions
```


## TODO

- [x] Add dataset files and download instructions
- [x] Complete README.md with detailed setup and usage instructions
- [ ] Implement inference functionality for arbitrary video inputs
- [x] Provide pretrained model weights
- [ ] Add documentation for dataset structure
- [x] Add documentation for running methods
- [ ] Add license for codes

## Citation

If you find this work useful, please consider citing our paper:

```bibtex
@misc{suzuki2025athleticsposeauthenticsportsmotion,
      title={AthleticsPose: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability},
      author={Tomohiro Suzuki and Ryota Tanaka and Calvin Yeung and Keisuke Fujii},
      year={2025},
      eprint={2507.12905},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2507.12905},
}
```

## License

The **AthleticsPose Dataset** is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License. See the following files for details:

- [LICENSE-DATASET](LICENSE-DATASET) - License for the dataset
- [LICENSES/CC-BY-NC-SA-4.0.txt](LICENSES/CC-BY-NC-SA-4.0.txt) - Full license text

Note: This license applies specifically to the dataset. The code in this repository may be licensed differently.

## Contact

For questions or issues, please open an issue on GitHub or contact the authors.
