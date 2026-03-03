# Day 10 — Invoice Price Prediction with ML.NET

**Part of:** 250-Day AI Engineer Journey
**Phase:** Phase 1 — Foundations
**Topic:** Hands-on Simple ML in .NET with ML.NET

## What This Project Does
Regression model that predicts invoice amounts
based on client tier, line items, service hours
and other invoice features using ML.NET.

## Concepts Demonstrated
- Regression vs Classification (predicting a number)
- Feature concatenation and normalisation
- SDCA Regression trainer
- Regression metrics — R², RMSE, MAE, MSE
- Multi-tier data handling (Small/Mid/Enterprise)
- Predicted vs Actual comparison table

## Tech Stack
- .NET 8.0
- ML.NET (Microsoft.ML)
- C#
- Visual Studio 2022

## How to Run
1. Clone the repository
2. Open InvoicePricePrediction.sln in Visual Studio 2022
3. Install NuGet: Microsoft.ML
4. Press F5 to run

## Sample Output
```
R² Score : 0.9821
RMSE     : ₹8,243
MAE      : ₹6,891

Invoice   : Enterprise | 20 items | 60 days | 65hrs
Predicted : ₹158,943
```

## Key Learning
ML.NET requires label columns to be named "Label"
via [ColumnName("Label")] attribute — learned this
by debugging a real schema mismatch error.

## Author
Adarsh | .NET Developer → Enterprise AI Engineer
250-Day Journey Started: February 2026