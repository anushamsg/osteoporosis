# osteoporosis

#  Osteoporosis Risk Prediction Using Machine Learning

## Overview

This project leverages machine learning techniques to predict the risk of osteoporosis based on individual lifestyle and demographic factors. By analyzing modifiable habits—such as calcium intake, physical activity, and alcohol consumption—the model aims to support early screening and preventive healthcare strategies.

* **Course**: Machine Learning in Health Sciences (HI 2453)
* **Institution**: University of Pittsburgh, School of Health and Rehabilitation Sciences
* **Author**: Sree Gayatri Anusha Mylavarapu
* **Instructor**: Dr. Leming Zhou, PhD, DSc

---

##  Objectives

* Predict the risk of osteoporosis using a dataset of 1,958 samples and 16 features.
* Identify lifestyle and demographic features that significantly impact bone health.
* Compare traditional ML models and deep learning techniques.
* Evaluate models using metrics like accuracy, precision, recall, F1 score, and ROC-AUC.

---

##  Motivation

* Osteoporosis affects over **200 million** people globally.
* Often diagnosed only after fractures occur due to limitations of DXA scans.
* Early lifestyle-based prediction is more cost-effective and widely accessible.
* Project aligns with **preventive care** goals to detect risks before clinical manifestation.

---

##  Dataset

* **Source**: Kaggle – [Lifestyle Factors Influencing Osteoporosis](https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis)
* **Size**: 1,958 records × 16 features
* **Target variable**: `Osteoporosis` (1 = Yes, 0 = No)

### Key Features:

* Demographics: Age, Gender, Ethnicity
* Lifestyle: Calcium/Vitamin D intake, Alcohol, Smoking, Physical activity
* Medical: Family history, Hormonal status, Medications

---

##  Methods

### Preprocessing

* One-Hot Encoding for categorical variables
* Missing value handling with `SimpleImputer`
* Feature scaling
* Feature selection based on statistical tests (T-tests)

### Models Used

* Random Forest (Best overall performance)
* Logistic Regression
* Support Vector Machine (SVM)
* Naive Bayes (Baseline model)
* Deep Learning (Improved recall)
* Gradient Boosting (Part of ensemble)

### Pipeline & Tuning

* Implemented with `sklearn.pipeline`
* Hyperparameter tuning via grid search
* Evaluation through 5-fold cross-validation

---

##  Results

| Model             | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| ----------------- | -------- | --------- | ------ | -------- | ------- |
| **Random Forest** | 89%      | 92%       | 80%    | 85%      | 0.88    |
| Deep Learning     | 88%      | 91%       | 78%    | 84%      | —       |
| Naive Bayes       | 81%      | 89%       | 70%    | 78%      | —       |

* **Best Recall**: Deep Learning
* **Most Balanced Model**: Random Forest

---

##  Confusion Matrix Summary

* **True Positives**: 146
* **False Negatives**: 53
* **Precision**: 94% (Few false alarms)
* **Recall**: 73% (Misses some cases—future optimization needed)

---

## Key Predictors (Feature Importance)

* Age (most significant)
* Calcium and Vitamin D intake
* Hormonal status
* Rheumatoid arthritis, hyperthyroidism
* Ethnicity and body weight

---

##  Limitations

* Lower recall in some models
* Dataset limited to 1,958 samples (not nationally representative)
* Deep learning requires more compute and regularization
* Some lifestyle factors may interact in complex ways not fully captured

---

##  Future Work

* Improve recall using **oversampling** or threshold tuning
* Introduce **stacking/blending** ensemble techniques
* Use **larger and diverse datasets**
* Explore **interpretable DL** like TabNet
* Add **time-to-event (survival) analysis**



##  References

* [Kaggle Dataset](https://www.kaggle.com/datasets/amitvkulkarni/lifestyle-factors-influencing-osteoporosis)
* [PubMed and NCBI Studies on Osteoporosis Risk Factors](https://www.ncbi.nlm.nih.gov/)
* [Journal of International Medical Research – Deep Learning for Osteoporosis](https://doi.org/10.1177/03000605241244754)
