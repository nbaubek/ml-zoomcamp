# Dataset

**Online Shoppers Purchasing Intention Dataset**
UCI Machine Learning Repository, ID 468
https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
DOI: [10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q)

- **Citation:** Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5F88Q
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) — confirmed on the UCI dataset page on 2026-11-11. Redistribution is allowed with credit, so the CSV is committed in this repo at `data/online_shoppers_intention.csv` (1 MB, 12,330 rows × 18 columns).
- **Class:** `Revenue` — 1,908 positives (15.47%), 10,422 negatives (84.53%).
- **Features:** 10 numeric + 8 categorical (of which `OperatingSystems`, `Browser`, `Region`, `TrafficType` are integer codes stored as numbers; `Weekend` and `Revenue` are booleans).
- **Known messiness handled in cleaning:** `Month` contains `June` spelled out while every other month is 3-letter; integer code columns must not be treated as quantities; no missing values.
