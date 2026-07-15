# MSCS 634 — Lab 2: KNN and RNN Classification

**Name:** Shuyang (J) Tang
**Course:** MSCS 634 — Advanced Data Mining
**Lab:** Lab 2 — K-Nearest Neighbors (KNN) and Radius Neighbors (RNN) Classification Using the Wine Dataset

---

## Purpose of the Lab

This lab explores how two instance-based classifiers — **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** — perform on the sklearn **Wine Dataset** (178 samples, 13 chemical features, 3 wine cultivar classes), and how their key hyperparameters affect classification accuracy:

- **KNN** was evaluated with k = 1, 5, 11, 15, 21
- **RNN** was evaluated with radius = 350, 400, 450, 500, 550, 600
- The dataset was split 80% training / 20% testing (stratified, `random_state=42`)
- Accuracy trends for both models were visualized and compared

## Repository Contents

| File | Description |
|---|---|
| `MSCS_634_Lab_2.ipynb` | Jupyter Notebook with all code, outputs, plots, and analysis |
| `README.md` | This file |

## Key Insights

**Results summary:**

| Model | Best accuracy | Best parameter | Trend |
|---|---|---|---|
| KNN | **0.8056** | k = 5 (flat through k = 21) | Rises from k = 1, then plateaus |
| RNN | 0.7222 | radius = 350 | Steadily decreases as radius grows |

1. **KNN outperformed RNN at every parameter value tested.** KNN automatically adapts to local data density (it always finds exactly k neighbors), while RNN's fixed radius behaves inconsistently across dense and sparse regions of the feature space.

2. **KNN was stable; RNN was parameter-sensitive.** KNN accuracy plateaued at 0.8056 from k = 5 onward. RNN accuracy dropped from 0.7222 (r = 350) to 0.6667 (r = 600) because larger radii admit increasingly distant points from other classes, diluting the local vote.

3. **k = 1 underperformed (0.7778)** — a single nearest neighbor is highly noise-sensitive; a slightly larger k smooths out noisy decisions.

4. **Feature scaling is the hidden story.** The Wine dataset's `proline` feature spans 278–1680 while most other features span 0–30, so `proline` dominates the Euclidean distance calculation. This is why meaningful radius values are in the hundreds. In practice, distance-based classifiers should be paired with feature standardization — a deliberate design choice in this lab was to follow the assignment's specified parameter ranges on unscaled data, which itself demonstrates the importance of scaling.

## Challenges Faced & Decisions Made

- **Radius interpretation:** The radius values (350–600) initially seemed unusually large for a dataset whose features are mostly single or double digits. Exploring the feature ranges revealed that the `proline` feature (278–1680) dominates Euclidean distances, which explains the scale of the radius values and became a core insight of the lab.
- **Stratified splitting:** I used `stratify=y` in the train/test split to preserve the class balance (59/71/48) in both sets, ensuring accuracy comparisons are not skewed by uneven class representation in the small (36-sample) test set.
- **Handling potential empty neighborhoods in RNN:** With small radii, RNN can encounter test points that have no training points within the sphere (which raises an error or requires an outlier label). At radius ≥ 350 on this dataset every test point had at least one neighbor, so no outlier handling was ultimately needed — but this limitation is discussed in the notebook as a key practical difference between RNN and KNN.
- **Reproducibility:** `random_state=42` is fixed so all reported accuracies are exactly reproducible.

## When to Prefer Each Model

- **KNN**: varying data density, guaranteed predictions needed, no natural distance threshold — the better choice for this dataset (k = 5, accuracy 0.8056).
- **RNN**: when a physically meaningful distance threshold exists, density is uniform, or abstaining on isolated points is desirable (e.g., outlier detection contexts).
