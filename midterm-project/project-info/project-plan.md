# ML Zoomcamp Midterm Project: Plan

This document describes the project: what it is, the decisions behind it, and the steps to build it. The dataset is not chosen yet, so dataset-specific parts are marked as **TBD**.

## 1. Goal

Build a complete, reproducible machine learning project on a tabular dataset, covering modules 1-6 of the course:

- Problem framing and the model selection process (module 1)
- Data preparation and EDA (modules 2, 3)
- Training and evaluating a classifier (modules 3, 4)
- Serving the model as a web service and packaging it in a container (module 5)
- Tree-based models and ensembles (module 6)

Out of scope for the midterm: Kubernetes and cloud deployment. A Dockerized FastAPI service is enough.

## 2. Key decisions and why

| Decision | Choice | Reason |
|---|---|---|
| Task type | Binary classification | Uses the most course modules at once (3, 4, 6). The decision threshold gives a real business choice to discuss. |
| Dataset | UCI Online Shoppers Purchasing Intention (ID 468) | 12,330 rows, mixed numeric/categorical, 15.5% positives, CC BY 4.0 license, not a course dataset. Details in `dataset-info.md`. |
| Models | Logistic regression, decision tree, random forest, XGBoost | Linear and tree-based models, all taught in the course. |
| Tuning | A small, documented parameter search for every model | Shows a real model selection process rather than a single default run. |
| Validation | Train / validation / test split, plus k-fold CV on the final candidates | Fits module 1, and CV makes the comparison more reliable on a moderately sized dataset. |
| Visualization | Seaborn and matplotlib | Accepted for the course. |
| Environment | `uv` with `pyproject.toml` and `uv.lock` | Reproducible dependencies and a virtual environment. |
| Serving | FastAPI with uvicorn | A lightweight web service for the model. |
| Container | Docker | Makes the service runnable anywhere with two commands. |

## 3. Business case template

> **Done:** the business case is written in `business-case.md` (purchase-intent scoring for an
> online shop; precision-vs-recall discussion; `PageValues` caveat). The template below is kept
> for reference.

Fill this in once the dataset is chosen. It becomes the problem description in the README.

1. **Who decides:** the person or team using the prediction (**TBD**).
2. **What action the prediction triggers:** for example, contact a customer, flag a case, or adjust a plan (**TBD**).
3. **What each error costs:**
   - False positive: acting when it was not needed (**TBD**).
   - False negative: missing a case that mattered (**TBD**).
4. **Metric that follows from the costs:** if misses are expensive, favor recall. If unnecessary actions are expensive, favor precision. Report ROC AUC as the overall ranking quality.
5. **Threshold:** choose it from the costs above, not by defaulting to 0.5.

## 4. Repository layout

```
project/
├── README.md
├── pyproject.toml
├── uv.lock
├── data/                  # dataset, or a download script plus instructions
├── notebook.ipynb
├── train.py
├── predict.py             # FastAPI app
├── Dockerfile
├── model.bin              # produced by train.py
└── docs/                  # screenshots of the service in use (optional)
```

## 5. Work phases

### Phase 1: Problem framing
Write the business case from section 3. Define the target column exactly (what counts as positive).

### Phase 2: Data
- Download the dataset and record its source and license.
- Commit it if the license and size allow. Otherwise provide a download script or exact link and where to place the file.
- Use relative paths only.

### Phase 3: Data cleaning
- **Standardize column names:** lowercase, with spaces replaced by underscores.
  ```python
  df.columns = df.columns.str.lower().str.replace(' ', '_')
  ```
- **Standardize categorical values the same way:** for every string column, lowercase and replace spaces with underscores, after stripping stray whitespace.
  ```python
  categorical = list(df.dtypes[df.dtypes == 'object'].index)
  for c in categorical:
      df[c] = df[c].str.lower().str.replace(' ', '_')
  ```
- Fix data types (numbers stored as text, dates, and so on).
- Handle missing values and obviously invalid values, and document each choice.
- Turn the target into a 0/1 column.

The same standardization must be applied in `train.py` and `predict.py`, otherwise incoming requests will not match the training data.

### Phase 4: EDA (Seaborn)
Cover each of these in the notebook:
- Shape, data types, and min/max values
- Missing values
- Target distribution (class balance)
- Distributions of numeric features
- Relationship between each feature and the target (for example, churn-style rate by category, or distributions split by target)
- Correlations between numeric features
- Feature importance, using at least one method: mutual information for categorical features, correlation for numeric features, or importance from a tree model

### Phase 5: Split

> **Amended by `class-imbalance-note.md`:** stratified 80/20 train/test split plus stratified
> 5-fold CV on the 80% replaces the 60/20/20 scheme (a 20% validation slice would hold only
> ~380 positives).

- Train / validation / test (for example 60 / 20 / 20) with a fixed random seed.
- Extract `y` as a NumPy array and delete the target column from the feature dataframes, to avoid target leakage.

### Phase 6: Model comparison and tuning
Train all four models and record results in one summary table:

| Model | Default score | Tuned parameters | Tuned score |
|---|---|---|---|
| Logistic regression | | `C` | |
| Decision tree | | `max_depth`, `min_samples_leaf` | |
| Random forest | | `n_estimators`, `max_depth`, `min_samples_leaf` | |
| XGBoost | | `eta`, `max_depth`, `min_child_weight`, number of rounds | |

Use ROC AUC on the validation set as the main comparison score. Use `DictVectorizer` (or an equivalent pipeline) for one-hot encoding.

### Phase 7: Evaluation
- ROC curve and ROC AUC
- Precision and recall across thresholds
- Choose the threshold according to the business case
- k-fold cross-validation for the top candidates, to confirm the ranking is stable

### Phase 8: Final model
- Retrain the best model with its chosen parameters on train + validation.
- Check it once on the test set. The score should be close to the validation score. If it is much worse, investigate leakage or overfitting.

### Phase 9: Scripts and service

**`train.py`**
- Loads the data and applies the same cleaning as the notebook.
- Trains the final model and saves it (for example with pickle, including the vectorizer).
- Prints a score so a reviewer can confirm it ran correctly.

**`predict.py`**
- Loads the saved model at startup.
- Defines a Pydantic model for the input fields.
- Exposes `POST /predict`, returning the probability and the decision at the chosen threshold.
- Optionally exposes `GET /health`.

**Environment**
- `uv init` / `uv add ...` to create `pyproject.toml` and `uv.lock`.
- Commit both files.

**`Dockerfile`**
- Slim Python base image.
- Install dependencies from `pyproject.toml` and `uv.lock` using uv with the frozen lockfile.
- Copy `predict.py` and the model file.
- Run uvicorn on `0.0.0.0` and expose the port.

### Phase 10: README
Write it last, and test it by following it on a fresh clone.

1. Problem description and how the solution will be used
2. Dataset: source, license, how to get it
3. Setup: `uv sync`, and how to activate the environment
4. Training: the `train.py` command
5. Running the service locally, with an example `curl` request
6. Docker: exact `docker build` and `docker run` commands
7. Results summary: best model and key metric

## 6. What reviewers look for

- **Problem description:** clear, with enough context to see how the solution will be used.
- **EDA:** value ranges, missing values, target analysis, and feature importance.
- **Model training:** several models, both linear and tree-based, with tuned parameters.
- **Script:** training logic exported to `train.py`.
- **Reproducibility:** the notebook and script run without errors, and the data is committed or easy to get.
- **Deployment:** the model is served through a web service.
- **Dependencies and environment:** a dependency file, a virtual environment, and README steps to install and activate it.
- **Containerization:** a Dockerfile plus README commands to build and run it.

## 7. Pre-submission checklist

- [ ] Fresh clone works by following only the README
- [ ] Notebook runs top to bottom without errors
- [ ] `train.py` reproduces the model without the notebook
- [ ] Docker image builds, runs, and answers a test request
- [ ] Column and value standardization is identical in notebook, `train.py`, and `predict.py`
- [ ] No target leakage: target removed from features, preprocessing fitted on training data only
- [ ] Random seeds are fixed
- [ ] `uv.lock` is committed

## 8. Common pitfalls

- Leakage through the target column or through preprocessing fitted on all the data.
- Tuning on the test set.
- Reporting accuracy alone on an imbalanced target.
- Different preprocessing in the notebook and in the scripts.
- A README that skips a step the author did without noticing.

## 9. Open items

- [x] Choose the domain and dataset — purchase-intent scoring; UCI Online Shoppers (ID 468)
- [x] Confirm the dataset license and size — CC BY 4.0; 12,330 rows, ~1 MB CSV
- [x] Write the business case (section 3) — see `business-case.md`
- [x] Decide whether to commit the data or provide a download script — commit the CSV (license allows it with credit; citation in README and `dataset-info.md`)