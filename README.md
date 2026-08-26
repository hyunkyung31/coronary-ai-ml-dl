# Coronary AI - ML & DL

관상동맥질환(Coronary Artery Disease, CAD)의 위험도를 평가하기 위해
**임상/혈액 데이터, 2D 관상동맥 조영술(XCA), 3D CT 데이터**를 활용한
Machine Learning / Deep Learning 모델을 개발하는 프로젝트입니다.

본 프로젝트에서는 각각의 데이터 유형을 이용한 **Unimodal 모델**을 구축하고,
최종적으로 영상과 임상 데이터를 결합한 **Multimodal AI 모델**을 비교·평가합니다.

---

## 1. Project Overview

본 프로젝트의 AI 연구 영역은 크게 세 가지로 구성됩니다.

| 영역            | 데이터            | 주요 목적                  |
| ------------- | -------------- | ---------------------- |
| Clinical ML   | 혈액검사 + 임상정보    | CAD 위험도 예측             |
| 2D Image AI   | AngioCAD XCA   | 관상동맥 영상 분석 및 CAD/협착 분류 |
| 3D Image AI   | COCA CT        | 관상동맥 석회화 분석            |
| Multimodal AI | XCA + Clinical | 영상과 임상정보를 결합한 CAD 예측   |

---

## 2. Overall AI Pipeline

```text
                    Patient Data
                         │
             ┌───────────┼───────────┐
             │           │           │
             ▼           ▼           ▼
        Clinical       2D XCA      3D CCTA
         Data          Image        Image
             │           │           │
             ▼           ▼           ▼
        Clinical ML   2D Image AI   3D Image AI
             │           │           │
             │           │           │
             └──────┐    │    ┌──────┘
                    │    │
                    ▼    ▼
                AI Evaluation
```

추가적으로 AngioCAD 데이터의 **XCA 영상과 Clinical Feature**를 결합하여
Multimodal AI 모델을 구축합니다.

---

# 3. Dataset

## 3.1 Clinical Dataset

### Z-Alizadeh Sani Extension Dataset

CAD 초기 위험도 평가를 위한 Clinical Machine Learning 모델에 사용합니다.

주요 변수는 다음과 같습니다.

* Demographic information
* Hypertension
* Diabetes Mellitus
* Smoking
* Blood pressure
* Blood laboratory results
* Lipid profile
* ECG
* Echocardiography
* Cardiovascular risk factors

### Purpose

```text
Clinical / Blood Features
          │
          ▼
     ML Model
          │
          ▼
     CAD Risk
```

### Candidate Models

* Logistic Regression
* Decision Tree
* Random Forest
* XGBoost
* MLP

---

# 4. 2D XCA Analysis

## Dataset

### AngioCAD

2D X-ray Coronary Angiography(XCA) 영상과
환자의 Clinical Feature를 포함한 데이터셋을 사용합니다.

본 데이터셋은 다음 두 가지 연구에 활용할 예정입니다.

### 1. Image Unimodal

```text
XCA Image
    │
    ▼
Image Encoder
    │
    ▼
Classification
```

XCA 영상만을 이용하여 CAD 및 관상동맥 협착 관련 분류를 수행합니다.

### Candidate Models

* ResNet50
* EfficientNet-B0/B2
* DenseNet121
* ConvNeXt-Tiny
* ViT
* Swin Transformer

---

## 4.1 XCA Segmentation

관상동맥은 복잡한 형태를 가지기 때문에
Object Detection뿐만 아니라 Segmentation 기반 혈관 분석을 실험합니다.

### Candidate Models

* YOLO Segmentation
* CNN-based Segmentation Models

### Preprocessing Experiments

다음 영상 전처리 방법을 비교합니다.

* Original
* CLAHE
* Gamma Correction
* CLAHE + Gamma
* Frangi Filter
* Black Hat
* ROI-based preprocessing

---

# 5. 3D Coronary CT Analysis

## Dataset

### COCA Dataset

3D Coronary CT 데이터를 이용하여
관상동맥 석회화(Coronary Artery Calcium)를 분석할 예정입니다.

COCA 데이터는 관상동맥 전체 협착 정도를 직접 평가하기보다는
**석회화 병변과 Coronary Artery Calcium 정보를 분석하는 방향**으로 활용합니다.

### Planned Pipeline

```text
CT / DICOM
     │
     ▼
3D Volume Preprocessing
     │
     ▼
Calcium Region Analysis
     │
     ▼
CAC / Calcium Burden
```

### Research Direction

* CT DICOM preprocessing
* Calcium lesion localization
* Calcium segmentation
* CAC-related feature extraction
* Calcium burden analysis
* 3D visualization

---

# 6. Clinical Unimodal Model

혈액검사 및 환자의 임상정보만을 이용하여
CAD 위험도를 예측하는 Machine Learning 모델을 구축합니다.

```text
Clinical Features
       │
       ▼
 ML / MLP Model
       │
       ▼
 CAD Prediction
```

### Candidate Models

| Model               | Role                           |
| ------------------- | ------------------------------ |
| Logistic Regression | Baseline                       |
| Decision Tree       | Interpretable ML               |
| Random Forest       | Ensemble Baseline              |
| XGBoost             | Main ML Candidate              |
| MLP                 | Deep Learning / Fusion Encoder |

---

# 7. Image Unimodal Model

AngioCAD의 XCA 영상만을 이용한 Deep Learning 모델입니다.

```text
XCA Image
    │
    ▼
Image Encoder
    │
    ▼
Feature Vector
    │
    ▼
Classifier
    │
    ▼
CAD / Stenosis Prediction
```

### Candidate Image Encoders

* ResNet50
* EfficientNet
* DenseNet121
* ConvNeXt
* Vision Transformer
* Swin Transformer

---

# 8. Multimodal AI

본 프로젝트에서는 단일 데이터 유형뿐만 아니라
**2D XCA 영상과 환자의 Clinical Feature를 결합한 Multimodal AI 모델**을 구축합니다.

Multimodal 모델에는 **AngioCAD 데이터셋 내부에서 동일 환자에 대응하는 영상과 임상정보**를 사용합니다.

## Input

```text
XCA Image
+
Clinical / Blood Features
```

## Architecture

```text
          XCA Image
              │
              ▼
        Image Encoder
              │
              ▼
        Image Features
              │
              │
              ├──────────────┐
                             │
      Clinical Features      │
              │              │
              ▼              │
        Clinical MLP         │
              │              │
              ▼              │
      Clinical Features      │
              │              │
              └───────┬──────┘
                      ▼
                    Fusion
                      │
                      ▼
                 Classifier
                      │
                      ▼
              CAD / Stenosis
```

---

## 8.1 Multimodal Input / Target

관상동맥 협착 정도와 관련된 일부 변수는
모델 입력 데이터가 아닌 **Target Label 생성용 데이터**로 사용합니다.

예:

* LM
* LAD
* LCX
* RCA
* Coronary stenosis information

따라서 기본적인 학습 구조는 다음과 같습니다.

```text
Input X
────────────────────────
XCA Image
Clinical Features


Target y
────────────────────────
CAD / Stenosis Label
```

---

# 9. Model Comparison

최종적으로 다음 세 가지 모델 구조를 비교합니다.

### Clinical Unimodal

```text
Clinical
   │
   ▼
ML / MLP
   │
   ▼
Prediction
```

### Image Unimodal

```text
XCA
 │
 ▼
CNN / Transformer
 │
 ▼
Prediction
```

### Multimodal

```text
XCA ─────── Image Encoder ──┐
                            │
                            ▼
                          Fusion
                            │
                            ▼
Clinical ─── Clinical MLP ──┘
                            │
                            ▼
                         Prediction
```

Multimodal 모델을 통해 영상만 사용하거나 임상정보만 사용했을 때와 비교하여
두 데이터 유형의 결합이 CAD 예측에 미치는 영향을 평가할 예정입니다.

---

# 10. Evaluation Metrics

## Classification

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Confusion Matrix

## Segmentation

* IoU
* Dice Score
* Precision
* Recall
* mAP

추후 모델 실험 완료 후 각 모델별 결과를 비교하여 정리할 예정입니다.

---

# 11. Repository Structure

```text
coronary-ai-ml-dl/
│
├── clinical/
│   ├── data/
│   ├── notebooks/
│   ├── preprocessing/
│   └── models/
│
├── xca_2d/
│   ├── data/
│   ├── preprocessing/
│   ├── classification/
│   └── segmentation/
│
├── coca_3d/
│   ├── preprocessing/
│   ├── calcium_analysis/
│   └── visualization/
│
├── multimodal/
│   ├── clinical_encoder/
│   ├── image_encoder/
│   ├── fusion/
│   └── training/
│
├── configs/
│
├── results/
│   ├── clinical/
│   ├── xca/
│   ├── coca/
│   └── multimodal/
│
├── utils/
│
├── requirements.txt
│
└── README.md
```

> 원본 의료영상 및 환자 관련 데이터는 GitHub Repository에 업로드하지 않습니다.

---

# 12. Tech Stack

### Programming

* Python

### Data Analysis

* Pandas
* NumPy
* Matplotlib

### Machine Learning

* Scikit-learn
* XGBoost
* LightGBM

### Deep Learning

* PyTorch
* Torchvision
* Ultralytics

### Medical Image Processing

* OpenCV
* pydicom
* SimpleITK
* Scikit-image

---

# 13. Project Status

### Clinical

* [ ] Dataset preprocessing
* [ ] Exploratory Data Analysis
* [ ] ML baseline
* [ ] Model comparison
* [ ] Final evaluation

### 2D XCA

* [ ] XCA preprocessing
* [ ] Segmentation experiments
* [ ] Image classification
* [ ] Model comparison
* [ ] Final evaluation

### 3D COCA

* [ ] DICOM preprocessing
* [ ] 3D volume reconstruction
* [ ] Calcium region analysis
* [ ] CAC feature analysis
* [ ] Visualization

### Multimodal

* [ ] Clinical encoder
* [ ] Image encoder
* [ ] Feature fusion
* [ ] Multimodal training
* [ ] Unimodal vs Multimodal comparison

---

# 14. Planned Experiments

본 프로젝트에서는 다음 실험을 순차적으로 진행합니다.

```text
Clinical ML
     │
     ├──────────────┐
     │              │
     ▼              ▼
Clinical Model    XCA Image Model
                     │
                     ▼
               Multimodal Fusion
                     │
                     ▼
             Performance Comparison


COCA
 │
 ▼
3D CT / Calcium Analysis
```

---

# 15. Results

현재 모델 학습 및 실험을 진행하고 있으며
세부 성능 결과는 실험 완료 후 업데이트할 예정입니다.

향후 다음 내용을 포함합니다.

* Clinical ML performance
* XCA classification performance
* XCA segmentation performance
* COCA calcium analysis results
* Multimodal model performance
* Unimodal vs Multimodal comparison

---

## Disclaimer

본 프로젝트는 연구 및 교육 목적으로 수행됩니다.

개발된 AI 모델 및 분석 결과는 실제 의료진의 진단이나
임상적 판단을 대체하기 위한 목적으로 사용되지 않습니다.
