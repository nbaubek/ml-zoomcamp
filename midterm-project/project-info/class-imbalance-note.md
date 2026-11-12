## Class imbalance and validation strategy

### The issue

The Online Shoppers dataset is imbalanced: about 84.5% of sessions end without a purchase and 15.5% end with one (10,422 negatives and 1,908 positives out of 12,330). This is moderate imbalance, not extreme, but it changes how the project is evaluated.

### Consequences and how to handle them

| Issue | What to do |
|---|---|
| **Accuracy is misleading.** Always predicting "no purchase" scores about 84.5% and is useless. | Use **ROC AUC** as the main comparison metric. Also report precision and recall. Show the class balance chart in the EDA. |
| **Splits can distort the class ratio.** A random split may leave one part with too few positives. | Use **stratified** splitting so each part keeps about 15.5% positives. |
| **A 0.5 threshold usually misses too many positives.** | Choose the decision threshold from the business costs (cost of a wasted discount vs. cost of a lost sale), using the precision/recall trade-off. |
| **Models may favor the majority class.** | Optionally try `class_weight='balanced'` (logistic regression, random forest) and `scale_pos_weight` (XGBoost). Compare scores with and without, and keep the weights only if they help. |
| **Resampling methods such as SMOTE.** | Not needed at this level of imbalance and not part of the course material. Skip them. |

### Validation strategy

With 12,330 rows, a single validation set would be small and noisy: 20% is about 2,466 rows with only about 380 positives. Training is fast, so cross-validation is affordable. Use this setup:

1. Hold out a **stratified 20% test set** (fixed random seed). Do not use it for any decision.
2. Run **stratified 5-fold cross-validation** on the remaining 80% to compare models and tune parameters. Use the mean ROC AUC across folds.
3. Retrain the best model once on the full 80% with its chosen parameters.
4. Check it **once** on the test set. The score should be close to the CV score.

### Rules that prevent leakage

- Fit scalers and encoders on the training folds only, then apply them to the held-out fold. Putting preprocessing and model in one `Pipeline` does this automatically.
- Save the fitted preprocessing together with the model so `predict.py` applies exactly the same transforms.
- Remove the target column from the feature dataframe right after splitting.

### What this changes in the plan

- **Phase 5 (Split):** stratified 80/20 train/test split instead of 60/20/20.
- **Phase 6 (Model comparison):** every model is scored with stratified 5-fold CV on the 80%.
- **Phase 7 (Evaluation):** threshold chosen from the business costs, using out-of-fold predictions.
- **Phase 8 (Final model):** retrain on the full 80%, then one test check.