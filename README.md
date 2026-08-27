# ML Lab

**Interactive Machine Learning Laboratory built entirely with Python.**

ML Lab is a terminal-based ML workstation: load a dataset, explore it, clean it,
engineer features, train and tune real scikit-learn models, evaluate them,
compare them, save/load them, track experiments, generate reports, and even
brush up on ML theory — all from one `python main.py` command, with a polished
`rich`-powered terminal UI.

This is a portfolio-grade project intended to demonstrate practical skills for
**Python Developer**, **Data Scientist**, and **Machine Learning / AI Engineer** roles.

---

## Overview

ML Lab is a Python-only command-line application. There is no web server, no
HTML/CSS/JS, and no browser frontend — everything runs directly in your
terminal via `python main.py`. Every result (accuracy, R², feature importance,
predictions, etc.) is computed for real using scikit-learn; nothing is faked
or hard-coded.

## Features

- **Dataset Studio** — load built-in datasets (Iris, House Prices, Customer
  Churn) or any custom CSV, with automatic column-type detection and target
  suggestions.
- **Exploratory Data Analysis** — histograms, box plots, scatter/line/bar/count
  plots, correlation heatmaps, all saved as PNGs.
- **Data Quality & Cleaning** — missing-value/duplicate/outlier/constant-column
  detection with recommendations, plus cleaning operations (imputation,
  de-duplication, dtype conversion, outlier capping/removal).
- **Feature Engineering & Selection** — log transforms, ratios, interactions,
  binning, polynomial features; variance threshold, correlation filtering,
  SelectKBest, and mutual information ranking.
- **Model Training** — 7 classification models, 7 regression models, 3
  clustering algorithms, all from a centralized model registry.
- **Hyperparameter Tuning** — GridSearchCV / RandomizedSearchCV with sensible,
  fast default search spaces.
- **Evaluation** — accuracy/precision/recall/F1/ROC-AUC, MAE/RMSE/R²/MAPE,
  silhouette score/inertia, confusion matrices, ROC & PR curves, residual
  plots, cross-validation, and baseline (Dummy model) comparison.
- **Model Comparison** — train and compare several models side-by-side on
  the same split.
- **Prediction** — enter feature values and get a real prediction (with
  confidence, for classifiers) from the currently trained model.
- **Experiment Tracking** — every training run is logged as JSON; list,
  view, compare, and delete experiments.
- **Model Manager** — save/load trained pipelines with Joblib + metadata.
- **Report Generator** — Markdown/text reports summarizing dataset, model,
  metrics, feature importance, and honest, non-hyperbolic conclusions.
- **Automated Pipeline** — one command runs load → validate → clean →
  preprocess → split → train → evaluate → compare → save → report.
- **ML Academy** — 15 short lessons (basics, overfitting, bias/variance,
  regularization, ensembles, etc.) each with a simple + technical explanation,
  example, common mistakes, and an interview question.
- **ML Quiz** — a local, offline question bank across Python, NumPy, Pandas,
  Statistics, ML, and Data Science at 3 difficulty levels.

## Architecture

```
ML-Lab/
├── main.py                 # Entry point (interactive + CLI modes)
├── config.py                # Centralized paths, defaults, AppState
├── ml_lab/
│   ├── ui/                  # rich-based menu, display, prompts
│   ├── data/                 # dataset manager, validator, cleaner, statistics
│   ├── preprocessing/        # ColumnTransformer pipeline, encoding, scaling,
│   │                          missing values, feature selection
│   ├── models/                # model registry, trainer, task accessors
│   ├── evaluation/            # classification/regression/clustering metrics,
│   │                          evaluator (CV, baseline, feature importance)
│   ├── tuning/                 # GridSearchCV / RandomizedSearchCV
│   ├── visualization/          # matplotlib/seaborn plot generation
│   ├── prediction/              # single-record prediction engine
│   ├── experiments/              # experiment tracker + history/comparison
│   ├── models_storage/            # Joblib model persistence
│   ├── reports/                    # Markdown/text report generator
│   ├── academy/                     # lessons + quiz question bank
│   └── pipeline/                     # automated end-to-end pipeline
├── datasets/                # iris.csv, house_prices.csv, customer_churn.csv
├── saved_models/             # *.joblib + *.json metadata (git-ignored)
├── experiments/                # experiment_NNN.json (git-ignored)
├── reports/                      # generated reports (git-ignored)
├── generated_plots/                # generated PNGs (git-ignored)
└── tests/                            # pytest suite
```

## Tech Stack

Python 3.11+, NumPy, Pandas, scikit-learn, SciPy, Matplotlib, Seaborn, Joblib,
Rich, pytest.

## Installation

```bash
git clone <your-repo-url>
cd ML-Lab
```

### Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### Requirements Installation

```bash
pip install -r requirements.txt
```

## Running the Application

```bash
python main.py
```

You'll see the ML Lab banner and main menu:

```
╔══════════════════════════════════════════════════════════╗
║                       ML LAB                              ║
║          Interactive Machine Learning Laboratory          ║
╚══════════════════════════════════════════════════════════╝
```

Navigate with the numbered menu options. Invalid input is always re-prompted
— the application never crashes on a bad keystroke.

## CLI Commands

```bash
python main.py                                   # Interactive mode
python main.py --dataset datasets/iris.csv        # Load a dataset on startup
python main.py --info datasets/iris.csv           # Print dataset info and exit
python main.py --pipeline datasets/iris.csv \
    --target species --problem-type classification # Run full pipeline and exit
```

## Dataset Usage

Three built-in datasets ship with the project:

| Dataset | Rows | Target | Problem type |
|---|---|---|---|
| Iris | 150 | `species` | Classification |
| House Prices (synthetic) | 800 | `price` | Regression |
| Customer Churn (synthetic) | 1,000 | `churn` | Classification |

You can also load **any CSV** via Dataset Studio → "Load custom CSV", or
`python main.py --dataset path/to/file.csv`.

## Supported Algorithms

**Classification:** Logistic Regression, KNN, Decision Tree, Random Forest,
Gradient Boosting, Gaussian Naive Bayes, SVM.

**Regression:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest,
Gradient Boosting, KNN Regressor.

**Clustering:** K-Means, DBSCAN, Agglomerative Clustering.

## Preprocessing

Numerical: `StandardScaler` / `MinMaxScaler` / `RobustScaler` with
`SimpleImputer`. Categorical: `OneHotEncoder` / `OrdinalEncoder` with
`SimpleImputer`. All wrapped in a scikit-learn `ColumnTransformer` +
`Pipeline` so preprocessing is **fit only on training data** — no leakage
into the test set.

## Model Evaluation

Real, calculated metrics only:
- Classification: accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
  classification report, ROC/PR curves.
- Regression: MAE, MSE, RMSE, R², MAPE, actual-vs-predicted & residual plots.
- Clustering: inertia, silhouette score, cluster sizes.
- Baseline comparison against `DummyClassifier` / `DummyRegressor`.
- K-Fold / Stratified K-Fold cross-validation.

## Hyperparameter Tuning

GridSearchCV or RandomizedSearchCV over small, sensible default search spaces
per model (defined in the model registry), so tuning stays fast on typical
demo-sized datasets.

## Experiment Tracking

Every training run (interactive or pipeline) logs an `experiment_NNN.json`
under `experiments/` with dataset, target, features, preprocessing, model,
parameters, metrics, and training time. Compare and delete experiments from
the Experiment Tracking menu.

## Model Saving

Trained pipelines are saved with Joblib (`saved_models/*.joblib`) alongside a
JSON metadata sidecar. **Security note:** `joblib.load` can execute arbitrary
code — only load `.joblib` files you trust (e.g. ones ML Lab itself saved).

## Report Generation

Generate a Markdown or text report summarizing dataset, model, metrics,
parameters, feature importance, and a factual conclusion (e.g. *"Random Forest
achieved the highest test F1 among the evaluated models"* — never
*"Random Forest is the best algorithm"*).

## Testing

```bash
pip install -r requirements.txt   # includes pytest
pytest
```

31 tests cover dataset loading, preprocessing/leakage safety, metrics
correctness, model training, experiment tracking, and the full automated
pipeline (classification + regression).

## Example Workflow

```
python main.py
1  → Dataset Studio → 1 → Load Iris
2  → Exploratory Data Analysis → 7 → Correlation heatmap
5  → Model Training → Random Forest → 80/20 split → Train
7  → Model Evaluation → confusion matrix, ROC curve, feature importance
9  → Prediction → enter sepal/petal measurements → get a real prediction
12 → Generate Report → report_*.md written to reports/
```

Or run the whole thing in one shot:

```bash
python main.py --pipeline datasets/customer_churn.csv --target churn --problem-type classification
```

## Troubleshooting

- **`ModuleNotFoundError`** — activate your virtual environment and re-run
  `pip install -r requirements.txt`.
- **Plots not appearing inline** — this is a terminal app; plots are saved as
  PNGs under `generated_plots/` and the file path is printed after each plot.
- **Custom CSV won't load** — check the error panel; it reports missing
  files, empty files, encoding issues, and malformed CSVs explicitly.
- **Windows paths** — use quotes around paths with spaces, e.g.
  `"C:\Users\You\Downloads\data.csv"`.

## Future Improvements

- Optional PDF report export.
- Additional ensemble/boosting libraries (XGBoost/LightGBM) as opt-in extras.
- Multi-target and time-series support.
- SHAP-based explainability alongside built-in feature importance.

## Important: GitHub Pages Clarification

**ML Lab is a Python CLI application and cannot be executed by GitHub Pages.**
GitHub Pages can only host static documentation (e.g. this README rendered as
a project landing page) — it cannot run Python. To actually use ML Lab, clone
the repository and run `python main.py` in a real Python environment
(locally, in a container, or on any machine with Python 3.11+).

```
GitHub Repository
        │
        ├── Python ML Lab  ──────────►  Runs locally / in a Python environment
        │
        └── README / Documentation ──►  Can be displayed via GitHub Pages
```

## License

MIT License — see `LICENSE` for details (add one if distributing publicly).
