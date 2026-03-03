using Microsoft.ML;
using Microsoft.ML.Data;


Console.WriteLine("=== Day 10 — ML.NET Sentiment Analysis ===\n");

var context = new MLContext(seed: 42);

// ── MORE training data (fixes the split problem) ──────────
var allData = new List<SentimentData>
{
    // Positive examples (12 total)
    new() { Text = "This product is amazing!",           Sentiment = true  },
    new() { Text = "Excellent service, very happy",      Sentiment = true  },
    new() { Text = "Great quality, will buy again",      Sentiment = true  },
    new() { Text = "Fast delivery, perfect item",        Sentiment = true  },
    new() { Text = "Absolutely love this product",       Sentiment = true  },
    new() { Text = "Best purchase I have ever made",     Sentiment = true  },
    new() { Text = "Outstanding quality and value",      Sentiment = true  },
    new() { Text = "Highly recommend to everyone",       Sentiment = true  },
    new() { Text = "Five stars, perfect in every way",   Sentiment = true  },
    new() { Text = "Exceeded all my expectations",       Sentiment = true  },
    new() { Text = "Super fast and works perfectly",     Sentiment = true  },
    new() { Text = "Fantastic product great value",      Sentiment = true  },

    // Negative examples (12 total)
    new() { Text = "Terrible product, avoid it",         Sentiment = false },
    new() { Text = "Very disappointed, poor quality",    Sentiment = false },
    new() { Text = "Waste of money, broken on day 1",    Sentiment = false },
    new() { Text = "Awful experience, never again",      Sentiment = false },
    new() { Text = "Does not work as described",         Sentiment = false },
    new() { Text = "Cheap quality, fell apart",          Sentiment = false },
    new() { Text = "Horrible, complete waste of money",  Sentiment = false },
    new() { Text = "Returned it, very bad product",      Sentiment = false },
    new() { Text = "One star, completely useless",       Sentiment = false },
    new() { Text = "Broke after one day, terrible",      Sentiment = false },
    new() { Text = "Do not buy this, total scam",        Sentiment = false },
    new() { Text = "Worst purchase I have ever made",    Sentiment = false },
};


// ── Manually split to GUARANTEE both classes in test ──────
// Take last 2 positive + last 2 negative as test set
var trainData = allData.Take(10)          // first 10 positive
    .Concat(allData.Skip(12).Take(10))    // first 10 negative
    .ToList();

var testData = allData.Skip(10).Take(2)   // last 2 positive
    .Concat(allData.Skip(22).Take(2))     // last 2 negative
    .ToList();

Console.WriteLine($"Training samples : {trainData.Count}");
Console.WriteLine($"Test samples     : {testData.Count}");
Console.WriteLine($"  Positive test  : {testData.Count(x => x.Sentiment)}");
Console.WriteLine($"  Negative test  : {testData.Count(x => !x.Sentiment)}");

// ── Load into IDataView ───────────────────────────────────
IDataView trainView = context.Data.LoadFromEnumerable(trainData);
IDataView testView = context.Data.LoadFromEnumerable(testData);

// ── Build pipeline ────────────────────────────────────────
var pipeline = context.Transforms.Text
    .FeaturizeText(
        outputColumnName: "Features",
        inputColumnName: nameof(SentimentData.Text))
    .Append(context.BinaryClassification.Trainers
        .SdcaLogisticRegression(
            labelColumnName: "Label",
            featureColumnName: "Features"));

// ── Train ─────────────────────────────────────────────────
Console.WriteLine("\nTraining model...");
ITransformer model = pipeline.Fit(trainView);
Console.WriteLine("Training complete!");

// ── Evaluate ──────────────────────────────────────────────
IDataView predictions = model.Transform(testView);
var metrics = context.BinaryClassification
    .Evaluate(predictions, labelColumnName: "Label");

Console.WriteLine("\n=== Model Evaluation ===");
Console.WriteLine($"Accuracy  : {metrics.Accuracy:P2}");
Console.WriteLine($"Precision : {metrics.PositivePrecision:P2}");
Console.WriteLine($"Recall    : {metrics.PositiveRecall:P2}");
Console.WriteLine($"F1 Score  : {metrics.F1Score:P2}");
Console.WriteLine($"AUC       : {metrics.AreaUnderRocCurve:P2}");

// ── Predictions ───────────────────────────────────────────
var predictor = context.Model
    .CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

var testSamples = new[]
{
    "This is the best thing I have bought this year",
    "Complete rubbish, do not buy this",
    "Decent product for the price",
    "Absolute garbage, fell apart immediately",
    "Really happy with my purchase",
    "Not what I expected at all",
};

Console.WriteLine("\n=== Live Predictions ===");
foreach (var text in testSamples)
{
    var result = predictor.Predict(
        new SentimentData { Text = text });

    var label = result.Prediction ? "POSITIVE ✅" : "NEGATIVE ❌";
    var confidence = result.Probability * 100;

    Console.WriteLine($"\nText      : {text}");
    Console.WriteLine($"Result    : {label}");
    Console.WriteLine($"Confidence: {confidence:F1}%");
}

// ── Save model ────────────────────────────────────────────
var modelPath = "sentiment_model.zip";
context.Model.Save(model, trainView.Schema, modelPath);
Console.WriteLine($"\n✅ Model saved → {modelPath}");

Console.WriteLine("\n=== Day 10 Complete! Press any key ===");
Console.ReadKey();


public class SentimentData
{
    [LoadColumn(0)]
    public string? Text { get; set; }

    [LoadColumn(1), ColumnName("Label")]
    public bool Sentiment { get; set; }
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")]
    public bool Prediction { get; set; }
    public float Probability { get; set; }
    public float Score { get; set; }
}
