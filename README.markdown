# Chest X-Ray Pneumonia Classification

This repository contains a Jupyter Notebook (`chest-xray-vision-98.ipynb`) for building a deep learning model to classify chest X-ray images as either "Pneumonia" or "Normal" using the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

## Project Overview

The notebook implements a convolutional neural network (CNN) using TensorFlow and Keras to detect pneumonia from chest X-ray images. It includes data preprocessing, model training, and evaluation steps. The dataset is organized into training, validation, and test sets, with images labeled as "PNEUMONIA" or "NORMAL."

### Key Features
- **Dataset**: Chest X-Ray Images (Pneumonia) dataset, containing 5,216 training images, 624 test images, and 16 validation images.
- **Preprocessing**: Images are loaded and organized into Pandas DataFrames for training, validation, and testing.
- **Model**: Utilizes TensorFlow/Keras with pre-trained models and custom layers for binary classification.
- **Libraries**: OpenCV, Pandas, NumPy, Seaborn, Matplotlib, TensorFlow, and Scikit-learn.
- **Hardware**: Optimized for GPU acceleration (CUDA libraries used, as seen in logs).

## Prerequisites

To run the notebook, ensure you have the following:

### Software
- Python 3.8+
- Jupyter Notebook or JupyterLab
- Kaggle API (for downloading the dataset)

### Python Libraries
Install the required libraries using:
```bash
pip install opencv-python pandas numpy seaborn matplotlib tensorflow scikit-learn
```

### Dataset
Download the dataset from Kaggle:
1. Install the Kaggle API: `pip install kaggle`
2. Configure your Kaggle API key (see [Kaggle API documentation](https://www.kaggle.com/docs/api)).
3. Download and unzip the dataset:
   ```bash
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
   unzip chest-xray-pneumonia.zip
   ```

The dataset should be placed in the project directory under `/kaggle/input/chest-xray-pneumonia/chest_xray/` (or adjust the paths in the notebook).

## Repository Structure

```
chest-xray-pneumonia-classification/
├── chest-xray-vision-98.ipynb  # Main Jupyter Notebook
├── README.md                   # This file
└── data/                       # Dataset directory (not included, download from Kaggle)
    └── chest_xray/
        ├── train/
        ├── test/
        └── val/
```

## Running the Notebook

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/chest-xray-pneumonia-classification.git
   cd chest-xray-pneumonia-classification
   ```

2. Ensure the dataset is in the correct directory (see "Dataset" section).

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

4. Open `chest-xray-vision-98.ipynb` and run the cells sequentially.

### Notes
- The notebook assumes a Kaggle-like environment with the dataset at `/kaggle/input/chest-xray-pneumonia/`. Modify the `train_dir`, `test_dir`, and `val_dir` paths if running locally.
- GPU support is recommended for faster training. The notebook includes CUDA-related logs, indicating GPU usage.
- If you encounter CUDA errors (e.g., `Unable to register cuFFT factory`), ensure your TensorFlow version is compatible with your CUDA/cuDNN setup.

## Results

The notebook processes the dataset and trains a model to classify chest X-rays. Performance metrics (e.g., accuracy, loss) are not explicitly shown in the provided code snippet but can be added during model evaluation.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Make changes and commit: `git commit -m "Add feature"`.
4. Push to your branch: `git push origin feature-name`.
5. Open a pull request.

## Acknowledgments

- Dataset provided by [Paul Timothy Mooney](https://www.kaggle.com/paultimothymooney) via Kaggle.
- Built with TensorFlow, Keras, and other open-source libraries.

For issues or questions, please open an issue on this repository.
