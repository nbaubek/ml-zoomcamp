## Business case: predicting purchase intent of online shoppers

### Problem

An online shop wants to know, while a visitor is still browsing, whether the session is likely to end in a purchase. Visitors who are close to buying are the best use of limited sales resources, but the shop cannot tell them apart from casual browsers by looking at them. The goal is to score each session with a purchase probability.

### Who uses the prediction

The e-commerce / conversion team of an online retailer.

### How the solution will be used

The model runs as a web service. The shop's site sends session statistics collected so far (pages visited, time spent, bounce and exit rates, traffic type, visitor type, month, and so on) and receives a purchase probability.

Sessions scoring above a chosen threshold are treated as high-intent and trigger an action, for example:
- opening a proactive live-chat window with a sales agent,
- or showing a small nudge such as free shipping at the right moment.

Sessions below the threshold are left alone.

### Why a threshold, not 0.5

Sales agents are a limited resource, and every unnecessary intervention has a cost. The threshold should be set from the cost of errors, not by default.

| Error | What happens | Cost (assumed, adjust to the shop) |
|---|---|---|
| **False positive** | An agent or offer is spent on a visitor who would not buy | Wasted agent time or discount margin |
| **False negative** | A high-intent visitor gets no help and leaves without buying | A lost sale |

If agent capacity is the bottleneck, favor **precision** (don't waste interventions). If a lost sale is much more expensive than an intervention, favor **recall**. The README should state which of the two it chose, pick the threshold from the precision/recall trade-off on out-of-fold predictions, and explain the reasoning.

### Target and metric

- **Target:** `revenue` (1 if the session ended with a purchase, 0 otherwise).
- **Class balance:** about 15.5% positive, so accuracy is misleading.
- **Main metric:** ROC AUC for comparing models, with precision and recall reported at the chosen threshold.

### Limitations to state honestly

- The model predicts who is likely to **buy**, not who would be **persuaded** by an intervention. A visitor who would have bought anyway still counts as a correct positive. Measuring the real effect of the intervention would need an A/B test, which is beyond this project.
- The dataset contains aggregated session statistics. Using it live assumes the same features can be computed partway through a session.
- `PageValues` is a Google Analytics metric computed from pages visited before a transaction. Check in the EDA whether it makes the problem artificially easy, and report results with and without it if it dominates.
- Data covers one retailer over one year, so performance may differ for other shops or seasons.

### Success criterion

A model that ranks sessions clearly better than chance (ROC AUC well above 0.5 on the held-out test set), with a documented threshold and a test score close to the cross-validation score.