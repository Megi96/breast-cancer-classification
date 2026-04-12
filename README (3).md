# 🔬 Breast Cancer Histopathology Classification Using Deep Learning

> AI-assisted classification of breast tumor subtypes from microscopic histopathology images using transfer learning and Vision Transformers.

[![HuggingFace Space](https://img.shields.io/badge/🤗%20Demo-Hugging%20Face%20Space-blue)](https://huggingface.co/spaces/Megi96/breakhis-demo)
[![Dataset](https://img.shields.io/badge/Dataset-BreaKHis-green)](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
[![Python](https://img.shields.io/badge/Python-3.10+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)

---

## 🌐 Live Demo

Try the interactive demo on Hugging Face Spaces — upload any breast histopathology image and get an instant classification with Attention Rollout visualization and clinical explanation:

**[👉 Launch Demo](https://huggingface.co/spaces/Megi96/breakhis-demo)**

---

## 📋 Project Overview

Manual histopathology analysis is time-consuming and requires expert pathologists. Current diagnostic workflows face significant bottlenecks due to limited specialist availability and increasing case volumes. This project builds a deep learning pipeline to classify 8 breast tumor subtypes from microscopic images with high accuracy and clinical relevance.

**Key innovation:** Leveraging Phikon-v2, a Vision Transformer pretrained specifically on histopathology images, to achieve superior performance over general-purpose CNN architectures — achieving **95.03% accuracy** on 8-class classification.

This project was developed by a histopathology lab technician supervisor, with a focus on clinical relevance, model interpretability, and real-world applicability.

---

## 📂 Repository Structure

```
breast-cancer-classification/
│
├── breakhis-01-eda.ipynb          # Exploratory data analysis & splits
├── breakhis-02-binary.ipynb       # Binary classification (benign vs malignant)
├── breakhis-03-multiclass.ipynb   # 8-class subtype classification
├── breakhis-04-gradcam.ipynb      # Attention Rollout & interpretability
│
├── app.py                         # Hugging Face Gradio demo
├── requirements.txt               # Python dependencies
└── README.md
```

---

## 🗃️ Dataset — BreaKHis

| Property | Value |
|---|---|
| Total images | 9,109 |
| Patients | 82 |
| Magnifications | 40X, 100X, 200X, 400X |
| Image size | 700 × 460 px, RGB, PNG |
| Benign samples | 2,480 |
| Malignant samples | 5,429 |
| Source | P&D Laboratory, Paraná, Brazil |

### 8 Tumor Classes

| Class | Category |
|---|---|
| Adenosis | 🟢 Benign |
| Fibroadenoma | 🟢 Benign |
| Phyllodes Tumor | 🟡 Benign* |
| Tubular Adenoma | 🟢 Benign |
| Ductal Carcinoma | 🔴 Malignant |
| Lobular Carcinoma | 🔴 Malignant |
| Mucinous Carcinoma | 🔴 Malignant |
| Papillary Carcinoma | 🔴 Malignant |

> \* Phyllodes tumor exists on a spectrum from benign to malignant. Clinical grading requires full histological assessment.

**Dataset challenge:** The dataset is notably imbalanced — ductal carcinoma dominates the malignant class, and the overall split is 69% malignant / 31% benign. Weighted loss functions and careful sampling strategies were used to address this.

---

## 🧪 Notebooks

### Notebook 01 — EDA (`breakhis-01-eda.ipynb`)
- Dataset loading and exploration
- Class distribution analysis (binary and multiclass)
- Magnification-level statistics
- Train/validation/test split creation — splits saved as CSV for reproducibility across notebooks

### Notebook 02 — Binary Classification (`breakhis-02-binary.ipynb`)
- Benign vs. malignant classification
- Three architectures compared: ResNet50, EfficientNet-B3, Phikon-v2
- Weighted loss functions for class imbalance
- Per-magnification performance analysis

### Notebook 03 — Multiclass Classification (`breakhis-03-multiclass.ipynb`)
- 8-class subtype classification
- Phikon-v2 backbone with fine-tuned MLP head
- Best checkpoint saved: `best_phikon_head.pth`
- Full evaluation: accuracy, macro F1, confusion matrix, classification report

### Notebook 04 — Attention Rollout & Interpretability (`breakhis-04-gradcam.ipynb`)
- Attention Rollout implementation for Phikon-v2 (ViT)
- Per-class attention gallery — one correctly predicted example per class
- Multi-magnification attention grid
- Confusion matrix and misclassification analysis
- Interactive live demo with clinical explanations

---

## 🏆 Model Comparison

| Model | Accuracy | Macro F1 | Pre-training |
|---|---|---|---|
| ResNet50 | 90.99% | 0.9141 | ImageNet (general) |
| EfficientNet-B3 | 59.48% | 0.5674 | ImageNet (general) |
| **Phikon-v2** | **95.03%** | **0.9456** | Histopathology (domain-specific) |

**Key insight:** Domain-specific pretraining outperforms general CNNs significantly. Phikon-v2's built-in understanding of nuclei structures, glandular formations, and stromal patterns gives it a decisive advantage over ImageNet-pretrained models.

---

## 🔭 Magnification Study

Performance was evaluated separately for each magnification level:

| Magnification | N | Accuracy | Macro F1 |
|---|---|---|---|
| **40X** | 315 | **96.51%** | **0.9576** |
| 100X | 291 | 94.85% | 0.9454 |
| 200X | 307 | 95.77% | 0.9523 |
| 400X | 274 | 92.70% | 0.9239 |

**Clinical implication:** 40X magnification provides the best performance. Lower magnification captures more contextual tissue architecture, enabling more accurate classification. For best results with the demo, use 40X images.

---

## 🧠 Model Architecture

**Backbone:** `owkin/phikon-v2` — Vision Transformer (ViT) pretrained on large-scale histopathology data by Owkin. 24 transformer layers, 1024-dimensional CLS token output.

**Classification head (fine-tuned):**
```
BatchNorm1d(1024)
→ Dropout(0.3)
→ Linear(1024 → 256)
→ ReLU()
→ Dropout(0.2)
→ Linear(256 → 8)
```

Only the head was trained — the backbone uses pretrained Owkin weights loaded directly from HuggingFace.

---

## 🔍 Interpretability — Attention Rollout

Unlike CNNs, Vision Transformers cannot use GradCAM (which requires convolutional feature maps). Instead, **Attention Rollout** is used — an algorithm that propagates self-attention weights across all 24 transformer layers to produce a spatial map of which image patches were most influential in the prediction.

**Reading the heatmap:**
- 🔴 Red / yellow = high attention (diagnostically relevant regions)
- 🔵 Blue = low attention (background tissue)

Attention maps were validated against pathologist knowledge — the model focuses on diagnostically relevant features such as nuclear pleomorphism, glandular architecture, and stromal patterns.

---

## 🌐 Hugging Face Demo

The interactive demo (`app.py`) is built with Gradio and deployed on Hugging Face Spaces. It includes:

- Image upload + 8-class prediction
- Attention Rollout heatmap visualization
- Confidence bar chart for all 8 classes
- Benign / malignant verdict with color coding
- Clinical explanation per predicted class (attended regions, histological patterns, clinical significance)
- Magnification selector with accuracy-aware warning
- Example images for all 8 classes

**[👉 Try it live](https://huggingface.co/spaces/Megi96/breakhis-demo)**

---

## ⚕️ Clinical Context

This project was developed from the perspective of a histopathology lab technician supervisor. The primary focus areas were:

- Verifying that model attention aligns with pathological hallmarks
- Establishing quality assurance principles for AI-assisted diagnosis
- Building explainability to support clinician trust and adoption
- Demonstrating viability for screening and workload reduction in resource-limited settings

**Potential applications:**
- High-throughput preliminary classification to triage cases for expert review
- Second-opinion validation tool for pathologists
- Educational tool for histopathology training

---

## ⚠️ Disclaimer

This tool is intended for **research and educational purposes only**. It is not validated for clinical use and must not be used for diagnostic decisions. Always consult a qualified pathologist for clinical interpretation.

---

## 🛠️ Setup

```bash
git clone https://github.com/Megi96/breast-cancer-classification
cd breast-cancer-classification
pip install -r requirements.txt
```

**Requirements:**
```
transformers
torch
torchvision
gradio
pillow
matplotlib
```

To run the demo locally:
```bash
python app.py
```

---

## 📧 Contact

**Author:** Megi Xibrraku
**Email:** xibrrakumegi@gmail.com
**GitHub:** [github.com/Megi96](https://github.com/Megi96)
**Demo:** [huggingface.co/spaces/Megi96/breakhis-demo](https://huggingface.co/spaces/Megi96/breakhis-demo)
**Dataset:** [BreaKHis — UFPR](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)

---

*Histopathology Lab · March 2026*
