# Day 10 — Sentiment Analysis with ML.NET

**Part of:** 250-Day AI Engineer Journey
**Phase:** Phase 1 — Foundations
**Topic:** Hands-on Simple ML in .NET with ML.NET

## What This Project Does
Binary classification model that predicts whether
a product review is Positive or Negative using ML.NET.

## Concepts Demonstrated
- MLContext setup and configuration
- IDataView data loading
- Text featurization (converting text → numbers)
- Binary classification with SDCA Logistic Regression
- Model evaluation — Accuracy, Precision, Recall, F1, AUC
- Saving and loading trained models

## Tech Stack
- .NET 8.0
- ML.NET (Microsoft.ML)
- C#
- Visual Studio 2022

## How to Run
1. Clone the repository
2. Open SentimentAnalysis.sln in Visual Studio 2022
3. Right-click project → Manage NuGet Packages
   → Install Microsoft.ML
4. Press F5 to run

## Sample Output
```
Training model...
Training complete!

=== Model Evaluation ===
Accuracy  : 100.00%
Precision : 100.00%
Recall    : 100.00%
F1 Score  : 100.00%
AUC       : 100.00%

Text      : This is the best thing I have bought
Result    : POSITIVE ✅
Confidence: 93.2%
```

## Key Learning
Small datasets (16 samples) cause class imbalance
issues in random train/test splits — learned to use
manual stratified splitting to guarantee both classes
appear in the test set.

## Author
Adarsh | .NET Developer → Enterprise AI Engineer
250-Day Journey Started: February 2026