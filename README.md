# AthleticsPose: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability (MMSports at ACMMM 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2507.12905-b31b1b.svg)](https://arxiv.org/abs/2507.12905)

![Dataset Example](docs/assets/all_pose_animation.gif)

## 🚧 Work in Progress

This repository is currently under development. Basic files and **Dataset** have been uploaded, but the project is still in preparation phase. Please check back later for updates.

## Abstract

This repository contains the implementation for our paper "AthleticsPose: Authentic Sports Motion Dataset on Athletic Field and Evaluation of Monocular 3D Pose Estimation Ability". We introduce a comprehensive dataset and evaluation framework for 3D pose estimation in athletics scenarios.

## Updates

- **2025-08-04**: Added dataset download instructions and license files
- **2025-07-18**: Initial repository setup and basic file upload

## Dataset Download

You can download the AthleticsPose dataset using either wget or curl:

### Using wget
```bash
wget -O AthleticsPoseDataset.zip \
  "https://github.com/SZucchini/AthleticsPose/releases/latest/download/AthleticsPoseDataset.zip"
```

### Using curl
```bash
curl -L -o AthleticsPoseDataset.zip \
  "https://github.com/SZucchini/AthleticsPose/releases/latest/download/AthleticsPoseDataset.zip"
```

After downloading, extract the dataset:
```bash
unzip AthleticsPoseDataset.zip
```

## TODO

- [x] Add dataset files and download instructions
- [ ] Complete README.md with detailed setup and usage instructions
- [ ] Implement inference functionality for arbitrary video inputs
- [ ] Provide pretrained model weights
- [ ] Add documentation for dataset structure and running methods

## Installation
Please use uv to install the package. This will ensure that all dependencies are correctly installed and the package is ready for use.

```bash
# Clone the repository
git clone https://github.com/your-username/AthleticsPose.git
cd AthleticsPose

# Install dependencies
uv sync
```

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
