# Used Cars — Linear & Logistic Regression (From Scratch)

A Machine Learning project implementing **Linear Regression** and **Logistic Regression**
from scratch using only NumPy, Pandas, and Matplotlib, applied to a real Used Cars dataset.
Includes full preprocessing, dimensionality reduction with PCA, and a comparison against
scikit-learn's implementations.

## 📊 Dataset

`Used_Cars.csv` — 4,340 used car listings with features such as brand, year, kilometers
driven, fuel type, transmission, owner history, and selling price.

- **Linear Regression Target:** `selling_price` (continuous)
- **Logistic Regression Target:** `Premium_car` (binary — 1 if price is above the 75th
  percentile of the dataset, else 0)

## 🧠 Project Structure

**Task 1 — Data Preprocessing**
- Handling missing data & duplicate rows
- Feature engineering (brand extraction, car age)
- Outlier removal (IQR method)
- Categorical encoding (label mapping + one-hot encoding)
- Z-score normalization
- PCA (retaining 95% of variance)

**Task 2 — Linear Regression (from scratch)**
- Model A: Baseline (no preprocessing)
- Model B: With normalization
- Model C: With normalization + PCA
- Evaluated with RMSE, R², and MAE

**Task 3 — Logistic Regression (from scratch)**
- Model A: Baseline (no preprocessing)
- Model B: With normalization
- Model C: With normalization + PCA
- Evaluated with Accuracy, Precision, Recall, and F1-Score
- Decision boundary visualization (RBF-kernel SVM, 2D projection)

**Task 4 — Overall Comparison**
- Performance summary across all 6 custom models
- Custom implementation vs. scikit-learn (`LinearRegression` / `LogisticRegression`)
- Discussion of preprocessing impact and custom-vs-library performance gaps

## 🛠️ Tools

`numpy` · `pandas` · `matplotlib` · `scikit-learn` (for comparison only) · `tqdm`

## 🎥 Video Walkthrough

- Part 1: https://youtu.be/ACWyCKRatuY
- Part 2: https://youtu.be/hWIpPtwhtlc

## 👩‍💻 Author

Basmala Fatooh — Machine Learning Lab, Final Project
