// Day 13 — Mini-Project: Classify Iris Dataset in C#

using Microsoft.ML;
using Microsoft.ML.Data;

// ── Data Classes ──────────────────────────────────────────



// ── Helper method — paste GetIrisData() from Step 2 here ──
// (place the full dataset method here)


Console.WriteLine("╔══════════════════════════════════════════════╗");
Console.WriteLine("║     Day 13 — Iris Species Classifier         ║");
Console.WriteLine("║     Mini-Project: ML.NET Multiclass          ║");
Console.WriteLine("╚══════════════════════════════════════════════╝\n");

var context = new MLContext(seed: 42);

// Load full 150-sample dataset
var allData = GetIrisData();

Console.WriteLine("── Dataset Summary ───────────────────────────");
Console.WriteLine($"Total samples    : {allData.Count}");
Console.WriteLine($"Iris-setosa      : {allData.Count(x => x.Species == "Iris-setosa")}");
Console.WriteLine($"Iris-versicolor  : {allData.Count(x => x.Species == "Iris-versicolor")}");
Console.WriteLine($"Iris-virginica   : {allData.Count(x => x.Species == "Iris-virginica")}");
Console.WriteLine($"Features         : SepalLength, SepalWidth, PetalLength, PetalWidth");

// ── Train / Test Split ────────────────────────────────────
// Take 40 from each species for training (120 total)
// Keep 10 from each species for testing (30 total)
var trainData = allData
    .Where(x => x.Species == "Iris-setosa").Take(40)
    .Concat(allData.Where(x => x.Species == "Iris-versicolor").Take(40))
    .Concat(allData.Where(x => x.Species == "Iris-virginica").Take(40))
    .ToList();

var testData = allData
    .Where(x => x.Species == "Iris-setosa").Skip(40)
    .Concat(allData.Where(x => x.Species == "Iris-versicolor").Skip(40))
    .Concat(allData.Where(x => x.Species == "Iris-virginica").Skip(40))
    .ToList();

Console.WriteLine($"\nTraining samples : {trainData.Count} (40 per species)");
Console.WriteLine($"Test samples     : {testData.Count}  (10 per species)");

IDataView trainView = context.Data.LoadFromEnumerable(trainData);
IDataView testView = context.Data.LoadFromEnumerable(testData);

// ── Build Pipeline ────────────────────────────────────────
// Step 1: Convert string label to key
// Step 2: Concatenate 4 numeric features
// Step 3: Normalise features (0-1 scale)
// Step 4: Train multiclass classifier
// Step 5: Convert key back to string label

var pipeline = context.Transforms.Conversion
    .MapValueToKey(
        outputColumnName: "Label",
        inputColumnName: nameof(IrisData.Species))

    .Append(context.Transforms.Concatenate("Features",
        nameof(IrisData.SepalLength),
        nameof(IrisData.SepalWidth),
        nameof(IrisData.PetalLength),
        nameof(IrisData.PetalWidth)))

    .Append(context.Transforms.NormalizeMinMax("Features"))

    .Append(context.MulticlassClassification.Trainers
        .SdcaMaximumEntropy(
            labelColumnName: "Label",
            featureColumnName: "Features"))

    .Append(context.Transforms.Conversion
        .MapKeyToValue("PredictedLabel"));

// ── Train ─────────────────────────────────────────────────
Console.WriteLine("\nTraining model...");
var watch = System.Diagnostics.Stopwatch.StartNew();
ITransformer model = pipeline.Fit(trainView);
watch.Stop();
Console.WriteLine($"Training complete in {watch.ElapsedMilliseconds}ms!");

// ── Evaluate ──────────────────────────────────────────────
IDataView predictions = model.Transform(testView);

var metrics = context.MulticlassClassification
    .Evaluate(predictions, labelColumnName: "Label");

Console.WriteLine("\n╔══════════════════════════════════════════════╗");
Console.WriteLine("║              Model Evaluation                ║");
Console.WriteLine("╚══════════════════════════════════════════════╝");
Console.WriteLine($"Macro Accuracy : {metrics.MacroAccuracy:P2}");
Console.WriteLine($"Micro Accuracy : {metrics.MicroAccuracy:P2}");
Console.WriteLine($"Log Loss       : {metrics.LogLoss:F4}");
Console.WriteLine($"Log Loss Red.  : {metrics.LogLossReduction:F4}");
Console.WriteLine($"  → Higher reduction = better model");

// ── Confusion Matrix ──────────────────────────────────────
Console.WriteLine("\n── Confusion Matrix ──────────────────────────");
Console.WriteLine($"{"",16} {"Pred Setosa",14} {"Pred Versicol",14} {"Pred Virginica",14}");
Console.WriteLine(new string('─', 60));

var cm = metrics.ConfusionMatrix;
var species = new[] { "Actual Setosa", "Actual Versicol", "Actual Virginica" };

for (int i = 0; i < cm.NumberOfClasses; i++)
{
    Console.Write($"{species[i],-16}");
    for (int j = 0; j < cm.NumberOfClasses; j++)
    {
        var val = (int)cm.Counts[i][j];
        var mark = i == j ? $"{val} ✅" : val > 0 ? $"{val} ❌" : $"{val}   ";
        Console.Write($"{mark,14}");
    }
    Console.WriteLine();
}

// ── Per-Species Accuracy ──────────────────────────────────
Console.WriteLine("\n── Per-Species Accuracy ──────────────────────");
var speciesNames = new[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" };
for (int i = 0; i < cm.NumberOfClasses; i++)
{
    var correct = (int)cm.Counts[i][i];
    var total = (int)cm.Counts[i].Sum();
    var pct = (double)correct / total * 100;
    Console.WriteLine($"{speciesNames[i],-20}: {correct}/{total} correct ({pct:F0}%)");
}

// ── Live Predictions ──────────────────────────────────────
var predictor = context.Model
    .CreatePredictionEngine<IrisData, IrisPrediction>(model);

Console.WriteLine("\n╔══════════════════════════════════════════════╗");
Console.WriteLine("║           New Flower Predictions             ║");
Console.WriteLine("╚══════════════════════════════════════════════╝");

var newFlowers = new[]
{
    new { Flower = new IrisData { SepalLength=5.1f, SepalWidth=3.5f, PetalLength=1.4f, PetalWidth=0.2f },
          Expected = "Iris-setosa",
          Note = "Small petals — typical setosa" },

    new { Flower = new IrisData { SepalLength=6.0f, SepalWidth=2.7f, PetalLength=5.1f, PetalWidth=1.6f },
          Expected = "Iris-versicolor",
          Note = "Medium petals — borderline case" },

    new { Flower = new IrisData { SepalLength=6.9f, SepalWidth=3.1f, PetalLength=5.4f, PetalWidth=2.1f },
          Expected = "Iris-virginica",
          Note = "Large petals — typical virginica" },

    new { Flower = new IrisData { SepalLength=5.8f, SepalWidth=2.7f, PetalLength=5.1f, PetalWidth=1.9f },
          Expected = "Iris-virginica",
          Note = "Edge case — could be versicolor" },

    new { Flower = new IrisData { SepalLength=4.6f, SepalWidth=3.6f, PetalLength=1.0f, PetalWidth=0.2f },
          Expected = "Iris-setosa",
          Note = "Very small — clearly setosa" },
};

Console.WriteLine($"\n{"Note",-32} {"Expected",-20} {"Predicted",-20} {"Match"}");
Console.WriteLine(new string('─', 80));

int correct2 = 0;
foreach (var item in newFlowers)
{
    var result = predictor.Predict(item.Flower);
    var isCorrect = result.PredictedSpecies == item.Expected;
    if (isCorrect) correct2++;

    Console.WriteLine(
        $"{item.Note,-32} {item.Expected,-20} {result.PredictedSpecies,-20} {(isCorrect ? "✅" : "❌")}");
}
Console.WriteLine($"\nNew flowers: {correct2}/{newFlowers.Length} correct");

// ── Trainer Comparison ────────────────────────────────────
Console.WriteLine("\n╔══════════════════════════════════════════════╗");
Console.WriteLine("║         Comparing Different Trainers         ║");
Console.WriteLine("╚══════════════════════════════════════════════╝");

var trainers = new (string Name, Func<IEstimator<ITransformer>> GetPipeline)[]
{
    ("SDCA MaxEntropy", () => context.Transforms.Conversion
        .MapValueToKey("Label", nameof(IrisData.Species))
        .Append(context.Transforms.Concatenate("Features",
            nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth),
            nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth)))
        .Append(context.Transforms.NormalizeMinMax("Features"))
        .Append(context.MulticlassClassification.Trainers
            .SdcaMaximumEntropy(labelColumnName: "Label"))
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))),

    ("LBFGS MaxEntropy", () => context.Transforms.Conversion
        .MapValueToKey("Label", nameof(IrisData.Species))
        .Append(context.Transforms.Concatenate("Features",
            nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth),
            nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth)))
        .Append(context.Transforms.NormalizeMinMax("Features"))
        .Append(context.MulticlassClassification.Trainers
            .LbfgsMaximumEntropy(labelColumnName: "Label"))
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))),

    ("NaiveBayes", () => context.Transforms.Conversion
        .MapValueToKey("Label", nameof(IrisData.Species))
        .Append(context.Transforms.Concatenate("Features",
            nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth),
            nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth)))
        .Append(context.Transforms.NormalizeMinMax("Features"))
        .Append(context.MulticlassClassification.Trainers
            .NaiveBayes(labelColumnName: "Label"))
        .Append(context.Transforms.Conversion.MapKeyToValue("PredictedLabel"))),
};

Console.WriteLine($"\n{"Trainer",-20} {"Macro Acc",12} {"Micro Acc",12} {"Time (ms)",12}");
Console.WriteLine(new string('─', 58));

foreach (var trainer in trainers)
{
    var sw = System.Diagnostics.Stopwatch.StartNew();
    var mdl = trainer.GetPipeline().Fit(trainView);
    sw.Stop();
    var preds = mdl.Transform(testView);
    var mets = context.MulticlassClassification
                        .Evaluate(preds, labelColumnName: "Label");

    Console.WriteLine(
        $"{trainer.Name,-20} {mets.MacroAccuracy,11:P1} {mets.MicroAccuracy,11:P1} {sw.ElapsedMilliseconds,11}ms");
}

// ── Save Best Model ───────────────────────────────────────
var modelPath = "iris_classifier_model.zip";
context.Model.Save(model, trainView.Schema, modelPath);
Console.WriteLine($"\n✅ Best model saved → {modelPath}");

Console.WriteLine("\n╔══════════════════════════════════════════════╗");
Console.WriteLine("║          Day 13 Mini-Project Complete!       ║");
Console.WriteLine("╚══════════════════════════════════════════════╝");
Console.WriteLine("Press any key to exit...");
Console.ReadKey();

// ── Dataset Method ────────────────────────────────────────


// Full 150-sample Iris dataset
// Source: UCI Machine Learning Repository
// Each row: SepalLength, SepalWidth, PetalLength, PetalWidth, Species

static List<IrisData> GetIrisData() => new()
{
    // ── Iris Setosa (50 samples) ───────────────────────
    new() { SepalLength=5.1f, SepalWidth=3.5f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.9f, SepalWidth=3.0f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.7f, SepalWidth=3.2f, PetalLength=1.3f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.6f, SepalWidth=3.1f, PetalLength=1.5f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.6f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.4f, SepalWidth=3.9f, PetalLength=1.7f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=4.6f, SepalWidth=3.4f, PetalLength=1.4f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.4f, PetalLength=1.5f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.4f, SepalWidth=2.9f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.9f, SepalWidth=3.1f, PetalLength=1.5f, PetalWidth=0.1f, Species="Iris-setosa" },
    new() { SepalLength=5.4f, SepalWidth=3.7f, PetalLength=1.5f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.8f, SepalWidth=3.4f, PetalLength=1.6f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.8f, SepalWidth=3.0f, PetalLength=1.4f, PetalWidth=0.1f, Species="Iris-setosa" },
    new() { SepalLength=4.3f, SepalWidth=3.0f, PetalLength=1.1f, PetalWidth=0.1f, Species="Iris-setosa" },
    new() { SepalLength=5.8f, SepalWidth=4.0f, PetalLength=1.2f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.7f, SepalWidth=4.4f, PetalLength=1.5f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=5.4f, SepalWidth=3.9f, PetalLength=1.3f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.5f, PetalLength=1.4f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=5.7f, SepalWidth=3.8f, PetalLength=1.7f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.8f, PetalLength=1.5f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=5.4f, SepalWidth=3.4f, PetalLength=1.7f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.7f, PetalLength=1.5f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=4.6f, SepalWidth=3.6f, PetalLength=1.0f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.3f, PetalLength=1.7f, PetalWidth=0.5f, Species="Iris-setosa" },
    new() { SepalLength=4.8f, SepalWidth=3.4f, PetalLength=1.9f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.0f, PetalLength=1.6f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.4f, PetalLength=1.6f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=5.2f, SepalWidth=3.5f, PetalLength=1.5f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.2f, SepalWidth=3.4f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.7f, SepalWidth=3.2f, PetalLength=1.6f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.8f, SepalWidth=3.1f, PetalLength=1.6f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.4f, SepalWidth=3.4f, PetalLength=1.5f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=5.2f, SepalWidth=4.1f, PetalLength=1.5f, PetalWidth=0.1f, Species="Iris-setosa" },
    new() { SepalLength=5.5f, SepalWidth=4.2f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.9f, SepalWidth=3.1f, PetalLength=1.5f, PetalWidth=0.1f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.2f, PetalLength=1.2f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.5f, SepalWidth=3.5f, PetalLength=1.3f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.9f, SepalWidth=3.1f, PetalLength=1.5f, PetalWidth=0.1f, Species="Iris-setosa" },
    new() { SepalLength=4.4f, SepalWidth=3.0f, PetalLength=1.3f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.4f, PetalLength=1.5f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.5f, PetalLength=1.3f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=4.5f, SepalWidth=2.3f, PetalLength=1.3f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=4.4f, SepalWidth=3.2f, PetalLength=1.3f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.5f, PetalLength=1.6f, PetalWidth=0.6f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.8f, PetalLength=1.9f, PetalWidth=0.4f, Species="Iris-setosa" },
    new() { SepalLength=4.8f, SepalWidth=3.0f, PetalLength=1.4f, PetalWidth=0.3f, Species="Iris-setosa" },
    new() { SepalLength=5.1f, SepalWidth=3.8f, PetalLength=1.6f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=4.6f, SepalWidth=3.2f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.3f, SepalWidth=3.7f, PetalLength=1.5f, PetalWidth=0.2f, Species="Iris-setosa" },
    new() { SepalLength=5.0f, SepalWidth=3.3f, PetalLength=1.4f, PetalWidth=0.2f, Species="Iris-setosa" },

    // ── Iris Versicolor (50 samples) ───────────────────
    new() { SepalLength=7.0f, SepalWidth=3.2f, PetalLength=4.7f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=6.4f, SepalWidth=3.2f, PetalLength=4.5f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=6.9f, SepalWidth=3.1f, PetalLength=4.9f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=5.5f, SepalWidth=2.3f, PetalLength=4.0f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=6.5f, SepalWidth=2.8f, PetalLength=4.6f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=5.7f, SepalWidth=2.8f, PetalLength=4.5f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=6.3f, SepalWidth=3.3f, PetalLength=4.7f, PetalWidth=1.6f, Species="Iris-versicolor" },
    new() { SepalLength=4.9f, SepalWidth=2.4f, PetalLength=3.3f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=6.6f, SepalWidth=2.9f, PetalLength=4.6f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=5.2f, SepalWidth=2.7f, PetalLength=3.9f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=5.0f, SepalWidth=2.0f, PetalLength=3.5f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=5.9f, SepalWidth=3.0f, PetalLength=4.2f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=6.0f, SepalWidth=2.2f, PetalLength=4.0f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=6.1f, SepalWidth=2.9f, PetalLength=4.7f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=5.6f, SepalWidth=2.9f, PetalLength=3.6f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=6.7f, SepalWidth=3.1f, PetalLength=4.4f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=5.6f, SepalWidth=3.0f, PetalLength=4.5f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=5.8f, SepalWidth=2.7f, PetalLength=4.1f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=6.2f, SepalWidth=2.2f, PetalLength=4.5f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=5.6f, SepalWidth=2.5f, PetalLength=3.9f, PetalWidth=1.1f, Species="Iris-versicolor" },
    new() { SepalLength=5.9f, SepalWidth=3.2f, PetalLength=4.8f, PetalWidth=1.8f, Species="Iris-versicolor" },
    new() { SepalLength=6.1f, SepalWidth=2.8f, PetalLength=4.0f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=6.3f, SepalWidth=2.5f, PetalLength=4.9f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=6.1f, SepalWidth=2.8f, PetalLength=4.7f, PetalWidth=1.2f, Species="Iris-versicolor" },
    new() { SepalLength=6.4f, SepalWidth=2.9f, PetalLength=4.3f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=6.6f, SepalWidth=3.0f, PetalLength=4.4f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=6.8f, SepalWidth=2.8f, PetalLength=4.8f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=6.7f, SepalWidth=3.0f, PetalLength=5.0f, PetalWidth=1.7f, Species="Iris-versicolor" },
    new() { SepalLength=6.0f, SepalWidth=2.9f, PetalLength=4.5f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=5.7f, SepalWidth=2.6f, PetalLength=3.5f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=5.5f, SepalWidth=2.4f, PetalLength=3.8f, PetalWidth=1.1f, Species="Iris-versicolor" },
    new() { SepalLength=5.5f, SepalWidth=2.4f, PetalLength=3.7f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=5.8f, SepalWidth=2.7f, PetalLength=3.9f, PetalWidth=1.2f, Species="Iris-versicolor" },
    new() { SepalLength=6.0f, SepalWidth=2.7f, PetalLength=5.1f, PetalWidth=1.6f, Species="Iris-versicolor" },
    new() { SepalLength=5.4f, SepalWidth=3.0f, PetalLength=4.5f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=6.0f, SepalWidth=3.4f, PetalLength=4.5f, PetalWidth=1.6f, Species="Iris-versicolor" },
    new() { SepalLength=6.7f, SepalWidth=3.1f, PetalLength=4.7f, PetalWidth=1.5f, Species="Iris-versicolor" },
    new() { SepalLength=6.3f, SepalWidth=2.3f, PetalLength=4.4f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=5.6f, SepalWidth=3.0f, PetalLength=4.1f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=5.5f, SepalWidth=2.5f, PetalLength=4.0f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=5.5f, SepalWidth=2.6f, PetalLength=4.4f, PetalWidth=1.2f, Species="Iris-versicolor" },
    new() { SepalLength=6.1f, SepalWidth=3.0f, PetalLength=4.6f, PetalWidth=1.4f, Species="Iris-versicolor" },
    new() { SepalLength=5.8f, SepalWidth=2.6f, PetalLength=4.0f, PetalWidth=1.2f, Species="Iris-versicolor" },
    new() { SepalLength=5.0f, SepalWidth=2.3f, PetalLength=3.3f, PetalWidth=1.0f, Species="Iris-versicolor" },
    new() { SepalLength=5.6f, SepalWidth=2.7f, PetalLength=4.2f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=5.7f, SepalWidth=3.0f, PetalLength=4.2f, PetalWidth=1.2f, Species="Iris-versicolor" },
    new() { SepalLength=5.7f, SepalWidth=2.9f, PetalLength=4.2f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=6.2f, SepalWidth=2.9f, PetalLength=4.3f, PetalWidth=1.3f, Species="Iris-versicolor" },
    new() { SepalLength=5.1f, SepalWidth=2.5f, PetalLength=3.0f, PetalWidth=1.1f, Species="Iris-versicolor" },
    new() { SepalLength=5.7f, SepalWidth=2.8f, PetalLength=4.1f, PetalWidth=1.3f, Species="Iris-versicolor" },

    // ── Iris Virginica (50 samples) ────────────────────
    new() { SepalLength=6.3f, SepalWidth=3.3f, PetalLength=6.0f, PetalWidth=2.5f, Species="Iris-virginica" },
    new() { SepalLength=5.8f, SepalWidth=2.7f, PetalLength=5.1f, PetalWidth=1.9f, Species="Iris-virginica" },
    new() { SepalLength=7.1f, SepalWidth=3.0f, PetalLength=5.9f, PetalWidth=2.1f, Species="Iris-virginica" },
    new() { SepalLength=6.3f, SepalWidth=2.9f, PetalLength=5.6f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.5f, SepalWidth=3.0f, PetalLength=5.8f, PetalWidth=2.2f, Species="Iris-virginica" },
    new() { SepalLength=7.6f, SepalWidth=3.0f, PetalLength=6.6f, PetalWidth=2.1f, Species="Iris-virginica" },
    new() { SepalLength=4.9f, SepalWidth=2.5f, PetalLength=4.5f, PetalWidth=1.7f, Species="Iris-virginica" },
    new() { SepalLength=7.3f, SepalWidth=2.9f, PetalLength=6.3f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.7f, SepalWidth=2.5f, PetalLength=5.8f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=7.2f, SepalWidth=3.6f, PetalLength=6.1f, PetalWidth=2.5f, Species="Iris-virginica" },
    new() { SepalLength=6.5f, SepalWidth=3.2f, PetalLength=5.1f, PetalWidth=2.0f, Species="Iris-virginica" },
    new() { SepalLength=6.4f, SepalWidth=2.7f, PetalLength=5.3f, PetalWidth=1.9f, Species="Iris-virginica" },
    new() { SepalLength=6.8f, SepalWidth=3.0f, PetalLength=5.5f, PetalWidth=2.1f, Species="Iris-virginica" },
    new() { SepalLength=5.7f, SepalWidth=2.5f, PetalLength=5.0f, PetalWidth=2.0f, Species="Iris-virginica" },
    new() { SepalLength=5.8f, SepalWidth=2.8f, PetalLength=5.1f, PetalWidth=2.4f, Species="Iris-virginica" },
    new() { SepalLength=6.4f, SepalWidth=3.2f, PetalLength=5.3f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=6.5f, SepalWidth=3.0f, PetalLength=5.5f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=7.7f, SepalWidth=3.8f, PetalLength=6.7f, PetalWidth=2.2f, Species="Iris-virginica" },
    new() { SepalLength=7.7f, SepalWidth=2.6f, PetalLength=6.9f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=6.0f, SepalWidth=2.2f, PetalLength=5.0f, PetalWidth=1.5f, Species="Iris-virginica" },
    new() { SepalLength=6.9f, SepalWidth=3.2f, PetalLength=5.7f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=5.6f, SepalWidth=2.8f, PetalLength=4.9f, PetalWidth=2.0f, Species="Iris-virginica" },
    new() { SepalLength=7.7f, SepalWidth=2.8f, PetalLength=6.7f, PetalWidth=2.0f, Species="Iris-virginica" },
    new() { SepalLength=6.3f, SepalWidth=2.7f, PetalLength=4.9f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.7f, SepalWidth=3.3f, PetalLength=5.7f, PetalWidth=2.1f, Species="Iris-virginica" },
    new() { SepalLength=7.2f, SepalWidth=3.2f, PetalLength=6.0f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.2f, SepalWidth=2.8f, PetalLength=4.8f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.1f, SepalWidth=3.0f, PetalLength=4.9f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.4f, SepalWidth=2.8f, PetalLength=5.6f, PetalWidth=2.1f, Species="Iris-virginica" },
    new() { SepalLength=7.2f, SepalWidth=3.0f, PetalLength=5.8f, PetalWidth=1.6f, Species="Iris-virginica" },
    new() { SepalLength=7.4f, SepalWidth=2.8f, PetalLength=6.1f, PetalWidth=1.9f, Species="Iris-virginica" },
    new() { SepalLength=7.9f, SepalWidth=3.8f, PetalLength=6.4f, PetalWidth=2.0f, Species="Iris-virginica" },
    new() { SepalLength=6.4f, SepalWidth=2.8f, PetalLength=5.6f, PetalWidth=2.2f, Species="Iris-virginica" },
    new() { SepalLength=6.3f, SepalWidth=2.8f, PetalLength=5.1f, PetalWidth=1.5f, Species="Iris-virginica" },
    new() { SepalLength=6.1f, SepalWidth=2.6f, PetalLength=5.6f, PetalWidth=1.4f, Species="Iris-virginica" },
    new() { SepalLength=7.7f, SepalWidth=3.0f, PetalLength=6.1f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=6.3f, SepalWidth=3.4f, PetalLength=5.6f, PetalWidth=2.4f, Species="Iris-virginica" },
    new() { SepalLength=6.4f, SepalWidth=3.1f, PetalLength=5.5f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.0f, SepalWidth=3.0f, PetalLength=4.8f, PetalWidth=1.8f, Species="Iris-virginica" },
    new() { SepalLength=6.9f, SepalWidth=3.1f, PetalLength=5.4f, PetalWidth=2.1f, Species="Iris-virginica" },
    new() { SepalLength=6.7f, SepalWidth=3.1f, PetalLength=5.6f, PetalWidth=2.4f, Species="Iris-virginica" },
    new() { SepalLength=6.9f, SepalWidth=3.1f, PetalLength=5.1f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=5.8f, SepalWidth=2.7f, PetalLength=5.1f, PetalWidth=1.9f, Species="Iris-virginica" },
    new() { SepalLength=6.8f, SepalWidth=3.2f, PetalLength=5.9f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=6.7f, SepalWidth=3.3f, PetalLength=5.7f, PetalWidth=2.5f, Species="Iris-virginica" },
    new() { SepalLength=6.7f, SepalWidth=3.0f, PetalLength=5.2f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=6.3f, SepalWidth=2.5f, PetalLength=5.0f, PetalWidth=1.9f, Species="Iris-virginica" },
    new() { SepalLength=6.5f, SepalWidth=3.0f, PetalLength=5.2f, PetalWidth=2.0f, Species="Iris-virginica" },
    new() { SepalLength=6.2f, SepalWidth=3.4f, PetalLength=5.4f, PetalWidth=2.3f, Species="Iris-virginica" },
    new() { SepalLength=5.9f, SepalWidth=3.0f, PetalLength=5.1f, PetalWidth=1.8f, Species="Iris-virginica" },
};

public class IrisData
{
    public float SepalLength { get; set; }
    public float SepalWidth { get; set; }
    public float PetalLength { get; set; }
    public float PetalWidth { get; set; }
    public string Species { get; set; } = string.Empty;
}


public class IrisPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedSpecies { get; set; } = string.Empty;
    public float[] Score { get; set; } = Array.Empty<float>();
}