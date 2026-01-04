# 🧬 Breast Cancer Classification Using Machine Learning & Neural Networks

## 📌 Introduction

Early and accurate detection of breast cancer is critical for improving patient survival rates. With the advancement of machine learning, predictive models can assist healthcare professionals by providing reliable diagnostic support based on medical data.

This project focuses on building and evaluating classification models to distinguish between **benign** and **malignant** breast tumors using structured diagnostic data. Both **traditional machine learning** and **deep learning** approaches are implemented to balance **model interpretability** and **prediction accuracy**.

---

## 🎯 Project Objectives

- Analyze diagnostic breast cancer data to extract meaningful patterns  
- Build reliable classification models with high sensitivity for malignant cases  
- Compare interpretable models with deep learning-based models  
- Evaluate performance using clinically relevant metrics  

---

## 📂 Dataset Information

- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset  
- **Source:** UCI Machine Learning Repository  
  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)

### Dataset Overview

- **Total Samples:** 569  
- **Features:** 30 numerical features  
- **Target Variable:**
  - `0` → Benign  
  - `1` → Malignant  

### Feature Description

The dataset contains features computed from digitized images of fine needle aspirates (FNA) of breast masses. These features describe characteristics of cell nuclei such as:

- Radius  
- Texture  
- Perimeter  
- Area  
- Smoothness  
- Compactness  
- Concavity  
- Symmetry  
- Fractal dimension  

---

## 🛠️ Methodology

### 1️⃣ Data Preprocessing

- Removed non-informative identifiers  
- Checked and validated missing values  
- Applied feature scaling using standardization  
- Used **scikit-learn Pipelines** for consistent preprocessing  
- Split data into training, validation, and test sets to prevent data leakage  

---

### 2️⃣ Model Development

#### 🔹 Decision Tree Classifier
- Provides model transparency and interpretability  
- Enables visualization of decision rules  
- Useful for understanding feature importance  

#### 🔹 Neural Network (Dense Architecture)
- Built using **TensorFlow / Keras**  
- Captures complex, non-linear feature interactions  
- Trained using backpropagation with optimized loss functions  

---

### 3️⃣ Model Evaluation

The following evaluation metrics were used:

- Accuracy  
- Precision  
- Recall (Sensitivity)  
- F1-score  

Models were evaluated on **separate validation and test datasets** to ensure generalization.

---

## 📊 Results

### Decision Tree Performance

| Metric      | Test Set | Validation Set |
|------------|----------|----------------|
| Accuracy   | 96.4%    | 100%           |
| Precision | 0.94     | 1.00           |
| Recall    | 1.00     | 1.00           |
| F1-score  | 0.97     | 1.00           |

### Neural Network Performance

- Achieved high predictive accuracy  
- Demonstrated stable training and validation curves  
- Minimal overfitting observed during training  

---

## 📈 Visualizations

- Decision Tree structure for interpretability  
- Neural network training and validation accuracy plots  
- Loss curves across epochs  

These visualizations help assess model behavior and training stability.

---

## 🧠 Key Insights

- Decision Trees offer strong performance with interpretability  
- Neural Networks capture deeper feature relationships  
- Feature scaling improves convergence and performance  
- High recall ensures malignant cases are rarely misclassified  
- Combining classical ML and deep learning yields balanced results  

---

## 🚀 How to Run the Project

### Requirements

```bash
pip install numpy pandas scikit-learn matplotlib tensorflow




## 🔮 Future Scope

- Implement k-fold cross-validation for more robust model evaluation  
- Add ensemble methods such as Random Forest and Gradient Boosting  
- Perform feature importance analysis and dimensionality reduction techniques  
- Deploy the trained model using Streamlit or Flask for interactive use  
- Extend the system to support real-time clinical decision-making  

---

## 📌 Conclusion

This project demonstrates the effectiveness of machine learning techniques in medical diagnostics. By combining interpretable models with deep learning approaches, the system achieves high reliability in breast cancer classification. Such methodologies have strong potential to assist clinicians, improve diagnostic accuracy, and support data-driven healthcare solutions.

