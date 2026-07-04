# 🛡️ Insurance Fraud Detection Pipeline

An end-to-end Machine Learning solution for identifying fraudulent insurance claims using advanced tabular data processing and tree-based ensemble models.

## 📌 Project Overview
This project provides a robust pipeline to detect fraudulent activities in insurance claims. It handles everything from raw data ingestion to model explainability. By leveraging **XGBoost** and **LightGBM**, combined with **SMOTE** for handling class imbalance, the system achieves high predictive performance even on highly skewed datasets.

## ✨ Key Features
- **Intelligent Preprocessing**: Automated handling of missing values, feature scaling, one-hot encoding for categorical variables, and auto-removal of identifier columns (e.g., `policy_number`, `claim_id`).
- **Label Normalization**: Built-in support for mapping various categorical labels (e.g., 'Y'/'N', 'Yes'/'No') to binary targets.
- **Imbalance Mitigation**: Integrated **SMOTE** (Synthetic Minority Over-sampling Technique) to address the classic "needle in a haystack" problem of fraud detection.
- **Dual-Model Approach**: Compares state-of-the-art Gradient Boosted Decision Trees (XGBoost vs. LightGBM).
- **Explainable AI (XAI)**: Generates **SHAP** (SHapley Additive exPlanations) plots to surface the most influential features behind a fraud prediction.
- **Modular Architecture**: Clean, modular code structure in `src/` for production-ready experimentation.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone <your-repo-url>
cd insurance-fraud-detection
pip install -r requirements.txt
```

### 2. Usage

#### Command Line Interface (CLI)
Run the full pipeline using `main.py`. This script will preprocess data, train models, evaluate them, and save artifacts.

```powershell
python main.py --data_path "data/Worksheet in Case Study question 2.csv" --smote --do_shap
```

> [!TIP]
> If running headlessly (e.g., on a remote server or CI pipeline), set the environment variable `MPLBACKEND=Agg` (e.g., prefix the command with `$env:MPLBACKEND='Agg';` in PowerShell or `MPLBACKEND=Agg` in Bash) to prevent Matplotlib's graphical interface from blocking execution.

**Arguments:**
- `--data_path`: Path to your CSV dataset.
- `--smote`: (Flag) Apply SMOTE to the training data.
- `--do_shap`: (Flag) Compute and save SHAP feature importance plots.
- `--out_dir`: Directory to save models (default: `models`).

#### Jupyter Notebook
For interactive analysis and step-by-step visualizations, use the provided notebook:
```bash
jupyter notebook insurance_fraud_detection.ipynb
```

## 📊 Results & Evaluation
The pipeline evaluates models using metrics crucial for imbalanced tasks:
- **ROC AUC**: Measures overall discriminative power.
- **PR-AUC (Average Precision)**: Best for fraud detection where the minority class is the focus.
- **F1-Score**: Balances Precision and Recall.

### Typical Performance (on sample data):
| Model | ROC AUC | PR-AUC | F1-Score |
| :--- | :--- | :--- | :--- |
| **XGBoost** | 0.8393 | 0.5749 | 0.6105 |
| **LightGBM** | 0.8307 | 0.5718 | 0.6667 |

## 📁 Project Structure
```text
├── data/               # Raw CSV datasets
├── models/             # Saved .joblib models and SHAP plots
├── src/                # Modular Python scripts
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── evaluate.py
│   └── utils.py
├── main.py             # Entry point for the pipeline
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🛠️ Requirements
- Python 3.8+
- Scikit-learn
- Imbalanced-learn
- XGBoost
- LightGBM
- SHAP
- Pandas, NumPy, Matplotlib

## 📝 License
This project is open-source and available for research and development purposes.
