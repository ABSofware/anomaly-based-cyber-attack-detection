# Anomaly-Based Cyber Attack Detection System

Professional interactive dashboard for anomaly-based network intrusion detection using K-Means and DBSCAN.

## Start in PyCharm / Windows

1. Open this folder in PyCharm.
2. Open the PyCharm Terminal.
3. Install dependencies:

```text
pip install -r requirements.txt
```

4. Run `main.py`.
5. The browser dashboard opens automatically at `http://127.0.0.1:8501`.

## Workflow

Upload training and testing CSV/TXT datasets → automatic dataset/label detection → preprocessing → choose K-Means, DBSCAN or Both → configure parameters → run real computation → inspect metrics/graphs → export CSV, JSON and PDF.

## Supported data

- NSL-KDD / KDD-style labelled datasets
- CICIDS-style labelled datasets
- Compatible generic labelled CSV/TXT network datasets

The application uses the existing attack label for evaluation. K-Means cluster IDs and DBSCAN cluster IDs are mapped to attack families using the majority-label mapping learned from the training set. DBSCAN has no native `predict` operation, so the application uses nearest core-point assignment within epsilon for the test set; points with no reachable core point are treated as anomalies/noise.

## Important

Results are computed from the uploaded data at runtime. No Chapter 3/Chapter 4 reference results are hard-coded into the application.

## DBSCAN resource safety and parameter recommendation

The application now performs a bounded DBSCAN preflight before execution. It estimates neighbourhood density from a sample of the preprocessed training data and can stop a configuration that is likely to require excessive memory. DBSCAN runs single-threaded (`n_jobs=1`) to reduce peak CPU/RAM pressure.

The application also computes a conservative data-derived starting recommendation:
- **epsilon (ε):** based on the 90th percentile of the k-distance distribution on a bounded sample;
- **MinPts:** bounded heuristic based on the number of model features;
- a higher **warning ε** is shown as a boundary for configurations that may become expensive.

The recommendation is a resource-safe starting point, not a guarantee of the best detection accuracy. The final experiment should still be evaluated using the reported metrics.
