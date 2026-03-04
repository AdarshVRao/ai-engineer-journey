// Program.cs — Support Ticket Priority Classifier
// Day 12 — Practice: Label Data, Train Simple Model

using Microsoft.ML;
using Microsoft.ML.Data;

// ── Main Program ──────────────────────────────────────────

Console.WriteLine("╔══════════════════════════════════════════════╗");
Console.WriteLine("║  Day 12 — Support Ticket Priority Classifier ║");
Console.WriteLine("║  Multiclass Classification with ML.NET       ║");
Console.WriteLine("╚══════════════════════════════════════════════╝\n");

var context = new MLContext(seed: 42);

// ── Your Labelled Dataset ─────────────────────────────────
var labelledData = new List<TicketData>
{
    // HIGH PRIORITY
    new() { Text = "System is completely down cannot process any invoices",          Priority = "HIGH"   },
    new() { Text = "Cannot login at all getting error on every attempt",             Priority = "HIGH"   },
    new() { Text = "Production database connection failed urgent",                   Priority = "HIGH"   },
    new() { Text = "All users locked out of the platform right now",                 Priority = "HIGH"   },
    new() { Text = "Invoice processing completely stopped blocking our operations",  Priority = "HIGH"   },
    new() { Text = "Data not saving losing all our work critical issue",             Priority = "HIGH"   },
    new() { Text = "Payment integration broken cannot complete transactions",        Priority = "HIGH"   },
    new() { Text = "Server error 500 on every page site is unusable",               Priority = "HIGH"   },
    new() { Text = "Urgent all reports showing wrong numbers financial close today", Priority = "HIGH"   },
    new() { Text = "Security breach suspected unauthorised access detected",         Priority = "HIGH"   },
    new() { Text = "API completely down our app cannot function at all",             Priority = "HIGH"   },
    new() { Text = "Database corrupted cannot access any historical invoices",       Priority = "HIGH"   },

    // MEDIUM PRIORITY
    new() { Text = "Export to PDF not working getting blank file",                  Priority = "MEDIUM" },
    new() { Text = "Dashboard loading very slowly takes 30 seconds",                Priority = "MEDIUM" },
    new() { Text = "Email notifications not being sent to clients",                 Priority = "MEDIUM" },
    new() { Text = "Some invoices showing incorrect tax calculation",               Priority = "MEDIUM" },
    new() { Text = "Search results not returning correct filtered data",            Priority = "MEDIUM" },
    new() { Text = "User permissions not saving after I update them",               Priority = "MEDIUM" },
    new() { Text = "Report shows different totals than invoice list",               Priority = "MEDIUM" },
    new() { Text = "Cannot upload files larger than 5MB getting error",             Priority = "MEDIUM" },
    new() { Text = "Calendar integration showing wrong timezone for meetings",      Priority = "MEDIUM" },
    new() { Text = "Bulk import failing for some records not all",                  Priority = "MEDIUM" },
    new() { Text = "Two factor authentication code not arriving by SMS",            Priority = "MEDIUM" },
    new() { Text = "Invoice template not applying custom branding correctly",       Priority = "MEDIUM" },

    // LOW PRIORITY
    new() { Text = "How do I export invoices to Excel format",                      Priority = "LOW"    },
    new() { Text = "Can you add dark mode to the dashboard please",                 Priority = "LOW"    },
    new() { Text = "Would like to request a new report for quarterly summary",      Priority = "LOW"    },
    new() { Text = "Where can I find the user guide documentation",                 Priority = "LOW"    },
    new() { Text = "Can the font size be made larger on invoice templates",         Priority = "LOW"    },
    new() { Text = "Is it possible to add custom fields to client profiles",        Priority = "LOW"    },
    new() { Text = "Logo looks slightly blurry on printed invoices cosmetic only",  Priority = "LOW"    },
    new() { Text = "How to set up automatic payment reminders",                     Priority = "LOW"    },
    new() { Text = "Suggestion to add colour coding to invoice status",             Priority = "LOW"    },
    new() { Text = "Would be helpful to have keyboard shortcuts",                   Priority = "LOW"    },
    new() { Text = "Can we get a mobile app version in future",                     Priority = "LOW"    },
    new() { Text = "Training video on how to use the bulk upload feature",          Priority = "LOW"    },
};

Console.WriteLine($"Total labelled samples : {labelledData.Count}");
Console.WriteLine($"HIGH   : {labelledData.Count(x => x.Priority == "HIGH")}");
Console.WriteLine($"MEDIUM : {labelledData.Count(x => x.Priority == "MEDIUM")}");
Console.WriteLine($"LOW    : {labelledData.Count(x => x.Priority == "LOW")}");

// ── Manual Split — guarantee all classes in test ──────────
var trainData = labelledData
    .Where(x => x.Priority == "HIGH").Take(10)
    .Concat(labelledData.Where(x => x.Priority == "MEDIUM").Take(10))
    .Concat(labelledData.Where(x => x.Priority == "LOW").Take(10))
    .ToList();

var testData = labelledData
    .Where(x => x.Priority == "HIGH").Skip(10)
    .Concat(labelledData.Where(x => x.Priority == "MEDIUM").Skip(10))
    .Concat(labelledData.Where(x => x.Priority == "LOW").Skip(10))
    .ToList();

Console.WriteLine($"\nTraining samples : {trainData.Count}");
Console.WriteLine($"Test samples     : {testData.Count}");

IDataView trainView = context.Data.LoadFromEnumerable(trainData);
IDataView testView = context.Data.LoadFromEnumerable(testData);

// ── Build Multiclass Pipeline ─────────────────────────────
var pipeline = context.Transforms.Conversion
    // Step 1: Convert string label to key (required for multiclass)
    .MapValueToKey(
        outputColumnName: "Label",
        inputColumnName: nameof(TicketData.Priority))

    // Step 2: Featurize the ticket text
    .Append(context.Transforms.Text.FeaturizeText(
        outputColumnName: "Features",
        inputColumnName: nameof(TicketData.Text)))

    // Step 3: Train multiclass classifier
    .Append(context.MulticlassClassification.Trainers
        .SdcaMaximumEntropy(
            labelColumnName: "Label",
            featureColumnName: "Features"))

    // Step 4: Convert prediction key back to string label
    .Append(context.Transforms.Conversion
        .MapKeyToValue("PredictedLabel"));

// ── Train ─────────────────────────────────────────────────
Console.WriteLine("\nTraining model on your labelled data...");
ITransformer model = pipeline.Fit(trainView);
Console.WriteLine("Training complete!");

// ── Evaluate ──────────────────────────────────────────────
IDataView predictions = model.Transform(testView);

var metrics = context.MulticlassClassification
    .Evaluate(predictions, labelColumnName: "Label");

Console.WriteLine("\n╔══════════════════════════════════════════════╗");
Console.WriteLine("║              Model Evaluation                ║");
Console.WriteLine("╚══════════════════════════════════════════════╝");
Console.WriteLine($"Macro Accuracy    : {metrics.MacroAccuracy:P2}");
Console.WriteLine($"Micro Accuracy    : {metrics.MicroAccuracy:P2}");
Console.WriteLine($"Log Loss          : {metrics.LogLoss:F4}");
Console.WriteLine($"  → Lower is better. 0 = perfect.");

// Per-class metrics
Console.WriteLine("\n── Per-Class Performance ─────────────────────");
Console.WriteLine($"{"Class",-10} {"Precision",12} {"Recall",10} {"F1",10}");
Console.WriteLine(new string('─', 46));

var confusionMatrix = metrics.ConfusionMatrix;
Console.WriteLine("\nConfusion Matrix:");
Console.WriteLine($"{"",14} {"Pred HIGH",12} {"Pred MED",12} {"Pred LOW",12}");
Console.WriteLine(new string('─', 52));

var classes = new[] { "HIGH", "MEDIUM", "LOW" };
for (int i = 0; i < confusionMatrix.NumberOfClasses; i++)
{
    Console.Write($"Actual {classes[i],-8}");
    for (int j = 0; j < confusionMatrix.NumberOfClasses; j++)
    {
        Console.Write($"{confusionMatrix.Counts[i][j],12}");
    }
    Console.WriteLine();
}

// ── Live Predictions ──────────────────────────────────────
var predictor = context.Model
    .CreatePredictionEngine<TicketData, TicketPrediction>(model);

Console.WriteLine("\n╔══════════════════════════════════════════════╗");
Console.WriteLine("║           New Ticket Predictions             ║");
Console.WriteLine("╚══════════════════════════════════════════════╝");

var newTickets = new[]
{
    // Should be HIGH
    "Everything is down we cannot do anything urgent help",
    // Should be MEDIUM
    "The invoice total is wrong it shows different amount",
    // Should be LOW
    "Please add an option to customise the colour theme",
    // Edge case — could be HIGH or MEDIUM
    "Some users cannot login but others can",
    // Edge case — ambiguous
    "Need help understanding how billing works",
};

foreach (var ticket in newTickets)
{
    var result = predictor.Predict(
        new TicketData { Text = ticket });

    Console.WriteLine($"\nTicket    : {ticket}");
    Console.WriteLine($"Priority  : {result.PredictedPriority}");
}

// ── Save Model ────────────────────────────────────────────
var modelPath = "ticket_classifier_model.zip";
context.Model.Save(model, trainView.Schema, modelPath);
Console.WriteLine($"\n✅ Model saved → {modelPath}");

Console.WriteLine("\n=== Day 12 Complete! Press any key ===");
Console.ReadKey();






// ── Data Classes ──────────────────────────────────────────

public class TicketData
{
    public string Text { get; set; } = string.Empty;
    public string Priority { get; set; } = string.Empty;
}

public class TicketPrediction
{
    [ColumnName("PredictedLabel")]
    public string PredictedPriority { get; set; } = string.Empty;
    public float[] Score { get; set; } = Array.Empty<float>();
}