# Student Intervention System

## Project Overview

This project, developed for the **Elements of Artificial Intelligence and Data Science** course (1st Year, 2nd Semester, 2024/2025), Assignment No. 2, aims to predict student pass/fail outcomes using the [UCI Student Performance dataset](https://archive.ics.uci.edu/ml/datasets/Student+Performance). The goal is to identify at-risk students (32.91% fail) for targeted interventions, supporting educational decision-making.

- **Dataset**: 395 students, 30 features (2 numerical: `age`, `absences`; 11 ordinal: e.g., `studytime`, `failures`; 17 categorical: e.g., `school`, `sex`), with binary target `passed` (`yes`=0, `no`=1).

- **Pipeline**:
  - **Exploratory Data Analysis (EDA)**: Analyzes feature distributions, class imbalance (67.09% pass), and key predictors (e.g., 88% fail for `failures` ≥1).
  - **Preprocessing**: Encodes features, caps `absences` outliers at 95th percentile (~20), applies SMOTE for imbalance, and selects top 20 features.
  - **Modeling**: Trains seven classifiers (Logistic Regression, Decision Tree, KNN, Random Forest, SVM, Neural Network, XGBoost) with GridSearchCV, optimizing for recall.
  - **Evaluation**: Assesses performance (e.g., Logistic Regression: 68% accuracy, 83% recall; XGBoost: ~70–75% accuracy) using accuracy, precision, recall, ROC/AUC, and visualizations.
  - **Interpretation**: Recommends interventions (e.g., tutoring for `failures` ≥1, attendance support for `absences` >10) based on feature importance (e.g., `failures`, `studytime`).

- **Deliverables**: Jupyter notebook (`student_intervention.ipynb`), due **May 30, 2025**, with a presentation scheduled for **May 26–30, 2025**.

- **Bonus Features**: SMOTE, feature selection, and XGBoost.

## Requirements

### Software
- **Python**: 3.11
- **Jupyter Notebook**: For running `student_intervention.ipynb`

### Libraries
Install required libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **matplotlib**, **seaborn**: Visualizations
- **scikit-learn**: Preprocessing, modeling, evaluation
- **imbalanced-learn**: SMOTE for class imbalance
- **xgboost**: XGBoost classifier

### Dataset
- **File**: `student-data.csv` (included, sourced from UCI)
- **Details**: 395 rows, 31 columns (30 features + `passed`). Ensure it’s in the project directory.
