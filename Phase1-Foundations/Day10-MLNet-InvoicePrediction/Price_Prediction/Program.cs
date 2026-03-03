using System;
using Microsoft.ML;
using Microsoft.ML.Data;

Console.WriteLine("╔══════════════════════════════════════════╗");
Console.WriteLine("║  Day 10 — Invoice Price Prediction       ║");
Console.WriteLine("║  ML.NET Regression Example               ║");
Console.WriteLine("╚══════════════════════════════════════════╝\n");


// Step 1: Create MLContext
var context = new MLContext(seed: 42);

// Step 2: Prepare training data
// Real-world: this comes from your SQL database
// Learning: inline data is fine
var invoiceData = new List<InvoiceData>
{
    // Small clients (Tier 1) — lower amounts
    new() { LineItems=3,  ClientTier=1, DaysToClose=15, ServiceHours=5,  IsRecurring=0, Amount=5000  },
    new() { LineItems=5,  ClientTier=1, DaysToClose=10, ServiceHours=8,  IsRecurring=1, Amount=8000  },
    new() { LineItems=2,  ClientTier=1, DaysToClose=20, ServiceHours=3,  IsRecurring=0, Amount=3000  },
    new() { LineItems=4,  ClientTier=1, DaysToClose=12, ServiceHours=6,  IsRecurring=1, Amount=7000  },
    new() { LineItems=6,  ClientTier=1, DaysToClose=18, ServiceHours=10, IsRecurring=0, Amount=9500  },
    new() { LineItems=3,  ClientTier=1, DaysToClose=8,  ServiceHours=4,  IsRecurring=1, Amount=6000  },
    new() { LineItems=7,  ClientTier=1, DaysToClose=25, ServiceHours=12, IsRecurring=0, Amount=11000 },
    new() { LineItems=2,  ClientTier=1, DaysToClose=14, ServiceHours=5,  IsRecurring=0, Amount=4500  },

    // Mid-size clients (Tier 2) — medium amounts
    new() { LineItems=8,  ClientTier=2, DaysToClose=20, ServiceHours=15, IsRecurring=1, Amount=25000 },
    new() { LineItems=10, ClientTier=2, DaysToClose=30, ServiceHours=20, IsRecurring=0, Amount=35000 },
    new() { LineItems=6,  ClientTier=2, DaysToClose=15, ServiceHours=12, IsRecurring=1, Amount=20000 },
    new() { LineItems=12, ClientTier=2, DaysToClose=25, ServiceHours=18, IsRecurring=0, Amount=40000 },
    new() { LineItems=9,  ClientTier=2, DaysToClose=22, ServiceHours=16, IsRecurring=1, Amount=30000 },
    new() { LineItems=7,  ClientTier=2, DaysToClose=18, ServiceHours=14, IsRecurring=0, Amount=22000 },
    new() { LineItems=11, ClientTier=2, DaysToClose=28, ServiceHours=22, IsRecurring=1, Amount=38000 },
    new() { LineItems=8,  ClientTier=2, DaysToClose=20, ServiceHours=16, IsRecurring=0, Amount=28000 },

    // Enterprise clients (Tier 3) — higher amounts
    new() { LineItems=15, ClientTier=3, DaysToClose=45, ServiceHours=40, IsRecurring=1, Amount=95000  },
    new() { LineItems=20, ClientTier=3, DaysToClose=60, ServiceHours=60, IsRecurring=0, Amount=150000 },
    new() { LineItems=12, ClientTier=3, DaysToClose=30, ServiceHours=35, IsRecurring=1, Amount=80000  },
    new() { LineItems=25, ClientTier=3, DaysToClose=90, ServiceHours=80, IsRecurring=0, Amount=200000 },
    new() { LineItems=18, ClientTier=3, DaysToClose=50, ServiceHours=50, IsRecurring=1, Amount=120000 },
    new() { LineItems=14, ClientTier=3, DaysToClose=40, ServiceHours=45, IsRecurring=0, Amount=100000 },
    new() { LineItems=22, ClientTier=3, DaysToClose=70, ServiceHours=70, IsRecurring=1, Amount=175000 },
    new() { LineItems=16, ClientTier=3, DaysToClose=55, ServiceHours=55, IsRecurring=0, Amount=130000 },
};

// Step 3: Load into IDataView
IDataView dataView = context.Data
    .LoadFromEnumerable(invoiceData);

// Step 4: Manual split — guarantee all tiers in both sets
// Take last 2 from each tier as test data
var trainData = invoiceData
    .Take(6)           // Tier 1 train (6 samples)
    .Concat(invoiceData.Skip(8).Take(6))   // Tier 2 train
    .Concat(invoiceData.Skip(16).Take(6))  // Tier 3 train
    .ToList();

var testData = invoiceData
    .Skip(6).Take(2)   // Tier 1 test (2 samples)
    .Concat(invoiceData.Skip(14).Take(2))  // Tier 2 test
    .Concat(invoiceData.Skip(22).Take(2))  // Tier 3 test
    .ToList();

IDataView trainView = context.Data.LoadFromEnumerable(trainData);
IDataView testView = context.Data.LoadFromEnumerable(testData);

Console.WriteLine($"Total samples    : {invoiceData.Count}");
Console.WriteLine($"Training samples : {trainData.Count}");
Console.WriteLine($"Test samples     : {testData.Count}");

// Step 5: Build regression pipeline
// Concatenate all feature columns → normalise → train
var pipeline = context.Transforms
    .Concatenate("Features",
        nameof(InvoiceData.LineItems),
        nameof(InvoiceData.ClientTier),
        nameof(InvoiceData.DaysToClose),
        nameof(InvoiceData.ServiceHours),
        nameof(InvoiceData.IsRecurring))
    .Append(context.Transforms
        .NormalizeMinMax("Features"))       // Scale all features 0-1
    .Append(context.Regression.Trainers
        .Sdca(
            labelColumnName: "Label",
            featureColumnName: "Features",
            maximumNumberOfIterations: 100));


// Step 6: Train model
Console.WriteLine("\nTraining regression model...");
ITransformer model = pipeline.Fit(trainView);
Console.WriteLine("Training complete!");

// Step 7: Evaluate on test data
IDataView predictions = model.Transform(testView);

var metrics = context.Regression
    .Evaluate(predictions,
        labelColumnName: "Label");

Console.WriteLine("\n╔══════════════════════════════════════════╗");
Console.WriteLine("║           Model Evaluation               ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine($"R² Score : {metrics.RSquared:F4}");
Console.WriteLine($"  → 1.0 = perfect | 0.0 = random | <0 = worse than random");
Console.WriteLine($"RMSE     : ₹{metrics.RootMeanSquaredError:N0}");
Console.WriteLine($"  → Average prediction error in rupees");
Console.WriteLine($"MAE      : ₹{metrics.MeanAbsoluteError:N0}");
Console.WriteLine($"  → Mean absolute error in rupees");
Console.WriteLine($"MSE      : {metrics.MeanSquaredError:N0}");
Console.WriteLine($"  → Mean squared error (penalises large errors)");

// Step 8: Make predictions on new invoices
var predictor = context.Model
    .CreatePredictionEngine<InvoiceData, InvoicePrediction>(model);

Console.WriteLine("\n╔══════════════════════════════════════════╗");
Console.WriteLine("║         New Invoice Predictions          ║");
Console.WriteLine("╚══════════════════════════════════════════╝");


var newInvoices = new[]
{
    new InvoiceData
    {
        LineItems    = 4,
        ClientTier   = 1,
        DaysToClose  = 14,
        ServiceHours = 7,
        IsRecurring  = 0,
        Amount       = 0   // Unknown — this is what we predict
    },
    new InvoiceData
    {
        LineItems    = 9,
        ClientTier   = 2,
        DaysToClose  = 22,
        ServiceHours = 17,
        IsRecurring  = 1,
        Amount       = 0
    },
    new InvoiceData
    {
        LineItems    = 20,
        ClientTier   = 3,
        DaysToClose  = 60,
        ServiceHours = 65,
        IsRecurring  = 0,
        Amount       = 0
    },
    new InvoiceData
    {
        LineItems    = 5,
        ClientTier   = 1,
        DaysToClose  = 10,
        ServiceHours = 9,
        IsRecurring  = 1,
        Amount       = 0
    },
};

// Labels for display
var invoiceLabels = new[]
{
    "Small client  | 4 items | 14 days | 7 hrs  | One-time",
    "Mid client    | 9 items | 22 days | 17 hrs | Recurring",
    "Enterprise    | 20 items| 60 days | 65 hrs | One-time",
    "Small client  | 5 items | 10 days | 9 hrs  | Recurring",
};

for (int i = 0; i < newInvoices.Length; i++)
{
    var result = predictor.Predict(newInvoices[i]);
    Console.WriteLine($"\nInvoice   : {invoiceLabels[i]}");
    Console.WriteLine($"Predicted : ₹{result.PredictedAmount:N0}");
}

// Step 9: Compare predictions vs actuals on test data
Console.WriteLine("\n╔══════════════════════════════════════════╗");
Console.WriteLine("║     Predicted vs Actual (Test Set)       ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine($"{"Tier",-8} {"Actual",12} {"Predicted",12} {"Error %",10}");
Console.WriteLine(new string('─', 46));

foreach (var invoice in testData)
{
    var result = predictor.Predict(invoice);
    var actual = invoice.Amount;
    var predicted = result.PredictedAmount;
    var errorPct = Math.Abs((predicted - actual) / actual) * 100;
    var tier = invoice.ClientTier == 1 ? "Small"
                  : invoice.ClientTier == 2 ? "Mid"
                  : "Enterprise";

    Console.WriteLine(
        $"{tier,-8} ₹{actual,10:N0} ₹{predicted,10:N0} {errorPct,8:F1}%");
}

// Step 10: Feature importance insight
Console.WriteLine("\n╔══════════════════════════════════════════╗");
Console.WriteLine("║           Key Insights                   ║");
Console.WriteLine("╚══════════════════════════════════════════╝");
Console.WriteLine("Features used to predict invoice amount:");
Console.WriteLine("  → LineItems    : More items = higher invoice");
Console.WriteLine("  → ClientTier   : Enterprise pays significantly more");
Console.WriteLine("  → DaysToClose  : Longer projects = larger invoices");
Console.WriteLine("  → ServiceHours : Direct correlation to amount");
Console.WriteLine("  → IsRecurring  : Recurring = slightly different pricing");

// Step 11: Save model
var modelPath = "invoice_prediction_model.zip";
context.Model.Save(model, trainView.Schema, modelPath);
Console.WriteLine($"\n✅ Model saved → {modelPath}");
Console.WriteLine("   Can be loaded in any .NET app — Web API, Worker, etc.");

Console.WriteLine("\n=== Day 10 Example 2 Complete! Press any key ===");
Console.ReadKey();

// Input — features the model learns from
public class InvoiceData
{
    [LoadColumn(0)]
    public float LineItems { get; set; }      // Number of line items

    [LoadColumn(1)]
    public float ClientTier { get; set; }     // 1=Small 2=Mid 3=Enterprise

    [LoadColumn(2)]
    public float DaysToClose { get; set; }    // Days to close the invoice

    [LoadColumn(3)]
    public float ServiceHours { get; set; }   // Hours of service billed

    [LoadColumn(4)]
    public float IsRecurring { get; set; }    // 1=Recurring 0=One-time

    [LoadColumn(5), ColumnName("Label")]
    public float Amount { get; set; }         // Invoice amount — what we PREDICT
}

// Output — what the model returns
public class InvoicePrediction
{
    [ColumnName("Score")]
    public float PredictedAmount { get; set; }
}

