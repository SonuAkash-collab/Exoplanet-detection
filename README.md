# 🪐 Kepler Exoplanet Classification Project

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-passing-brightgreen?logo=scikit-learn&logoColor=white)

This project uses the Kepler KOI (Kepler Objects of Interest) dataset to build a robust machine learning pipeline that classifies planetary candidates as either **Confirmed Exoplanets** or **False Positives**.

The primary notebook (`02_exoplanets_clean_preprocess.ipynb`) demonstrates an end-to-end workflow from data cleaning and preprocessing to model training, comparison, and explainability.

## ✨ Key Features

* **Robust Preprocessing:** Implements a full `scikit-learn` pipeline for imputation (median/mode), outlier capping (IQR), scaling (`RobustScaler`), and one-hot encoding.
* **Target Engineering:** Filters out "CANDIDATE" rows to create a clean, binary classification problem (Confirmed vs. False Positive).
* **Model Comparison:** Trains and evaluates four different models: LightGBM, RandomForest, XGBoost, and a simple MLP (Neural Network).
* **Explainability (XAI):** Generates SHAP summary plots and feature importance charts to understand the LightGBM model's decisions.
* **Artifact Generation:** Saves the final processed dataset (`.csv`/`.parquet`) and the fitted `preprocess_pipeline.pkl` for easy inference.

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/kepler-exoplanet-classifier.git](https://github.com/YourUsername/kepler-exoplanet-classifier.git)
    cd kepler-exoplanet-classifier
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Get the data:**
    Download the `exoplanets.csv` dataset (e.g., from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/)) and place it in the root folder. The notebook is configured to look for it there.

4.  **Run the notebook:**
    Launch Jupyter and run the `02_exoplanets_clean_preprocess.ipynb` notebook from top to bottom.

## 📊 Results

The model comparison showed that tree-based ensemble methods performed exceptionally well. The **RandomForest** model achieved the highest performance on the holdout test set.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **RandomForest** | **0.9931** | **0.9873** | **0.9915** | **0.9894** | **0.9995** | **0.9989** |
| MLP | 0.9903 | 0.9811 | 0.9894 | 0.9852 | 0.9991 | 0.9980 |
| XGBoost | 0.9917 | 0.9832 | 0.9915 | 0.9873 | 0.9989 | 0.9969 |
| LightGBM | 0.9917 | 0.9852 | 0.9894 | 0.9873 | 0.9986 | 0.9902 |

*Metrics based on the 20% holdout test set.*

## 🔮 Future Work

Based on the project's findings, potential next steps include:
* Hyperparameter tuning (e.g., with Optuna or GridSearch)
* Probability calibration for the best-performing model
* Deeper feature engineering and leakage checks
* Packaging the pipeline and model into a lightweight API (like FastAPI) for deployment

