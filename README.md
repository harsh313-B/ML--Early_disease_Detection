# Cardiovascular Disease Prediction

A machine learning classification project that predicts the presence of disease in patients using clinical measurements, health indicators, and lifestyle attributes.

---

## Dataset Overview[cite: 4]

* **Records:** 70,000 patient entries (`Data_file - 3.csv`)[cite: 4]
* **Target Variable:** `disease` (Binary classification: `0` = No Disease, `1` = Disease)[cite: 4]
* **Features Included:** Patient demographics, cholesterol levels, blood pressure/clinical metrics, and lifestyle factors (with non-predictive `id` and `date` fields removed during preprocessing)[cite: 4].

---

## Project Workflow[cite: 4]

1. **Exploratory Data Analysis (EDA):**[cite: 4]
   * Visualized the target class balance (`disease` distribution)[cite: 4].
   * Examined `cholesterol` levels across healthy vs. diseased patient groups using boxplots[cite: 4].
   * Generated a correlation heatmap across all numeric features to inspect feature relationships[cite: 4].
2. **Data Preprocessing:**[cite: 4]
   * Dropped identifier and timestamp columns (`id`, `date`)[cite: 4].
   * Encoded categorical object columns into numeric labels using `LabelEncoder`[cite: 4].
   * Split the dataset into training (80% — 56,000 rows) and testing (20% — 14,000 rows) sets with `random_state=42`[cite: 4].
   * Standardized feature scales using `StandardScaler` fit on the training set[cite: 4].
3. **Model Training & Evaluation:**[cite: 4]
   * Trained a **Logistic Regression** classifier (`max_iter=2000`)[cite: 4].
   * Trained a **Decision Tree** classifier (`max_depth=5`, `random_state=42`)[cite: 4].
   * Trained a **Linear Support Vector Machine (SVM)** (`kernel='linear'`)[cite: 4].
   * Evaluated all three models using Accuracy, Precision, Recall, and F1 Score[cite: 4].

---

## Model Performance Comparison[cite: 4]

| Model | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: |
| **Decision Tree (`max_depth=5`)**[cite: 4] | **73.46%**[cite: 4] | 76.33%[cite: 4] | **68.14%**[cite: 4] | **72.00%**[cite: 4] |
| **Linear SVM**[cite: 4] | 72.56%[cite: 4] | **77.21%**[cite: 4] | 64.16%[cite: 4] | 70.08%[cite: 4] |
| **Logistic Regression**[cite: 4] | 72.31%[cite: 4] | 74.48%[cite: 4] | 68.01%[cite: 4] | 71.10%[cite: 4] |

### Key Conclusion[cite: 4]
The **Decision Tree** model achieved the strongest overall performance with **73.46% accuracy**, **68.14% recall**, and an **F1 Score of 72.00%**, outperforming both Logistic Regression and Linear SVM as the primary model for early disease detection on this dataset[cite: 4].

---

## Tech Stack & Libraries[cite: 4]

* **Python 3.10**[cite: 4]
* **Data Manipulation:** `pandas`, `numpy`[cite: 4]
* **Visualization:** `matplotlib`, `seaborn`[cite: 4]
* **Machine Learning:** `scikit-learn`[cite: 4]

---

## How to Run

1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn jupyter
