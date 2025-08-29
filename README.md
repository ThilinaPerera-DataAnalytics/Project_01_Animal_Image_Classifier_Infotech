# 🐾 Animal Image Classifier – Deep Learning with PyTorch

An advanced deep learning project that classifies animal images using **Convolutional Neural Networks (CNNs)** and **transfer learning** with PyTorch.

This repository demonstrates the full pipeline of building an end-to-end image classification system; from dataset preparation, transformations, model training, evaluation, and visualization of predictions.

---

## 🚀 Features

* ✅ **End-to-End Workflow**: Data preprocessing, training, evaluation, and deployment-ready pipeline.
* ✅ **Transfer Learning**: Fine-tuning pre-trained CNNs for improved accuracy.
* ✅ **GPU Acceleration**: Optimized training with CUDA/cuDNN when available.
* ✅ **Visualization**: Model predictions with sample outputs for interpretability.
* ✅ **Custom Dataset Support**: Easily adaptable to new image datasets.

---

## 📂 Project Structure

```
Project_01_Animal_Image_Classifier_Infotech/
├── 1_data/
│   ├── test/                  # Images for final model testing
│   ├── train/                 # Images for training (e.g., cats, dogs, snakes subfolders)
│   │   ├── cats/
│   │   ├── dogs/
│   │   └── snakes/
│   └── val/                   # Images for validation during training (e.g., cats, dogs, snakes subfolders)
│       ├── cats/
│       ├── dogs/
│       └── snakes/
|
├── 2_models/                  # Stores trained PyTorch model files (.pth)
│   ├── best_model.pth         # Model with best validation accuracy
│   └── final_model.pth        # Model after the last training epoch
|
├── 3_notebooks/               # Contains the Jupyter Notebook/Python scripts
│   └── your_project_notebook.ipynb
|
├── 4_results/                 # Stores evaluation reports and confusion matrices
│   ├── val_classification_report.txt
│   └── val_confusion_matrix.png
|
|── 5_outputs/                 # Stores visual outputs
|    ├── 1. Predictions/       # Optional: For saving individual prediction images
|    └── 2. Graphs/            # For saving training and validation graphs (e.g., loss, accuracy curves)
|
|── ReadMe.md                  # Official documentation
└── LICENSE.md
```

---

## 🛠️ Tech Stack

* **Language**: Python
* **Deep Learning Framework**: PyTorch
* **Libraries**: Torchvision, Matplotlib, NumPy, PIL
* **Environment**: Jupyter Notebook
* **IDE**: Visual Studio Code

---

## ⚙️ Installation & Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/ThilinaPerera-DataAnalytics/Project_01_Animal_Image_Classifier_Infotech.git
   cd Project_01_Animal_Image_Classifier_Infotech
   ```

2. **Install dependencies**
    ```
    Ensure you have all the required libraries by running:

    pip install -r requirements.txt
    
    (torch, numpy, torchvision, matplotlib, time, os, PIL, tempfile, scikitlearn, seaborn, pandas)
    ```

3.  **Prepare the Dataset**
    ```
    Download the dataset from the Kaggle link provided above and place the `train`, `val`, and `test` subdirectories into the `1_data` folder as per the folder structure.
    ```

4.  **Run the Jupyter Notebook**
    ```
    Open the `your_project_notebook.ipynb` (or similar name) located in the `3_notebooks` directory using Jupyter Notebook or JupyterLab and execute the cells sequentially.
    ```

<font size='4'>The notebook will</font>

    * Load and preprocess images.
    * Initialize and train two models: a fine-tuned ResNet-18 model and a ResNet-18 feature extractor.
    * Save the trained models in the `2_models` directory.
    * Visualize model predictions.
    * Generate and save classification reports and confusion matrices.
    * Display training and validation graphs.

---

## 📊 Workflow Overview

1. **Data Transformations** – Normalize and augment datasets for robust training.
2. **Dataset Loading** – Structured image dataset using `ImageFolder`.
3. **Model Architecture** – Pre-trained CNN models fine-tuned for classification.
4. **Training Loop** – Custom training function with optimizer & learning rate scheduler.
5. **Evaluation** – Accuracy and loss tracking per epoch.
6. **Visualization** – Display predictions on validation images.

## Model Performance and Evaluation

The project evaluates the performance of both the fine-tuned CNN model (`model_ft`) and the transfer learning feature extraction model (`model_conv`) on the validation dataset.

The evaluation includes:
* **Loss and Accuracy per Epoch:** Printed during the training process for both training and validation phases.
* **Classification Report:** Provides precision, recall, f1-score, and support for each class.
* **Confusion Matrix:** Visualizes the number of correct and incorrect predictions for each class.

---

## 🖼️ Sample Output

| Input Image | Predicted Class |
| ----------- | --------------- |
| 🐶 Dog      | ✅ Dog           |
| 🐱 Cat      | ✅ Cat           |
| 🐍 Snake    | ✅ Snake         |

---

## 📌 Next Steps

* 🔹 Experiment with **ResNet, VGG, and EfficientNet** backbones.
* 🔹 Deploy the trained model with **Flask/FastAPI** for inference.
* 🔹 Improve performance with **hyperparameter tuning** & **data augmentation**.
* 🔹 Add **Grad-CAM visualization** for interpretability.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo and submit a pull request with improvements.

---

## 👤 Author

**Thilina Perera**

🔗 [LinkedIn](https://www.linkedin.com/in/thilina-perera-148aa934/) | [GitHub](https://github.com/ThilinaPerera-DataAnalytics)

---

## ⭐ Acknowledgements

* [PyTorch](https://pytorch.org/) – Deep Learning Framework
* [Torchvision](https://pytorch.org/vision/stable/index.html) – Image Utilities
* [Matplotlib](https://matplotlib.org/) – Visualization
* [Kaggle: Animal Image Classification Dataset](https://www.kaggle.com/datasets/borhanitrash/animal-image-classification-dataset/data) - Dataset Source - Train and test
* Validation images - Google images
---

✨ If you like this project, don’t forget to **star ⭐ the repo**!

---
