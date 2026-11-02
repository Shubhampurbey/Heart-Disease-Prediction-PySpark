"""
Model Inference and Test Prediction Pipeline for Cardiovascular Disease Prediction.

This script executes Task 9:
1. Loads the untouched preprocessed test Parquet dataset (13,697 records).
2. Loads the trained Logistic Regression and Random Forest models from models/.
3. Generates class predictions and calibrated probability scores for every test record.
4. Formats predictions with:
   - patient_id (stable patient tracking)
   - actual_class (0 or 1)
   - predicted_class (0 or 1)
   - probability_no_disease (P(cardio=0))
   - probability_disease (P(cardio=1))
   - model_name ('Logistic Regression' or 'Random Forest')
5. Exports predictions to results/predictions/test_predictions/ in Parquet format.
6. Verifies exact record counts, valid probability boundaries [0, 1], and binary class integrity.
"""

import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from pyspark.ml.classification import (
    LogisticRegressionModel,
    RandomForestClassificationModel
)


def get_spark_session(app_name: str = "CardioModelEvaluation") -> SparkSession:
    """Initialize and return an optimized PySpark SparkSession."""
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def load_test_dataset(spark: SparkSession, test_path: str):
    """Load test Parquet dataset."""
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")
    df = spark.read.parquet(test_path)
    count = df.count()
    print(f"[+] Loaded test dataset from: {test_path} | Count: {count:,}")
    assert count == 13697, f"Expected 13,697 records in test dataset, found {count}"
    return df, count


def load_trained_models(models_dir: str):
    """Load persisted Logistic Regression and Random Forest models."""
    lr_dir = os.path.join(models_dir, "logistic_regression")
    rf_dir = os.path.join(models_dir, "random_forest")

    if not os.path.exists(lr_dir):
        raise FileNotFoundError(f"Logistic Regression model not found at: {lr_dir}")
    if not os.path.exists(rf_dir):
        raise FileNotFoundError(f"Random Forest model not found at: {rf_dir}")

    print(f"[+] Loading Logistic Regression model from: {lr_dir}")
    lr_model = LogisticRegressionModel.load(lr_dir)

    print(f"[+] Loading Random Forest model from: {rf_dir}")
    rf_model = RandomForestClassificationModel.load(rf_dir)

    return lr_model, rf_model


def generate_predictions_for_model(model, test_df, model_name: str, features_col: str):
    """
    Generate predictions and extract structured probabilities.
    Returns DataFrame with:
    patient_id, actual_class, predicted_class, probability_no_disease, probability_disease, model_name
    """
    print(f"[+] Generating predictions using {model_name}...")
    raw_preds = model.transform(test_df)

    get_prob_0 = F.udf(lambda v: round(float(v[0]), 5), DoubleType())
    get_prob_1 = F.udf(lambda v: round(float(v[1]), 5), DoubleType())

    formatted_df = raw_preds.select(
        F.col("patient_id").cast(IntegerType()),
        F.col("cardio").cast(IntegerType()).alias("actual_class"),
        F.col("prediction").cast(IntegerType()).alias("predicted_class"),
        get_prob_0(F.col("probability")).alias("probability_no_disease"),
        get_prob_1(F.col("probability")).alias("probability_disease"),
        F.lit(model_name).alias("model_name")
    )
    return formatted_df


def save_test_predictions(combined_preds_df, output_dir: str):
    """Save combined test predictions in Parquet format and CSV format for reporting."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[+] Writing predictions to Parquet at: {output_dir}")
    (
        combined_preds_df
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(output_dir)
    )
    print("[SUCCESS] Test predictions Parquet saved successfully.")


def verify_test_predictions(spark: SparkSession, predictions_dir: str, expected_per_model: int):
    """
    Task 11: Verify prediction record counts, probability ranges, and class values.
    """
    print("\n" + "=" * 80)
    print("VERIFYING TEST PREDICTIONS DATASET")
    print("=" * 80)

    reloaded_df = spark.read.parquet(predictions_dir)
    total_reloaded = reloaded_df.count()
    expected_total = expected_per_model * 2

    print(f"[+] Total Reloaded Predictions : {total_reloaded:,} (Expected: {expected_total:,})")
    assert total_reloaded == expected_total, f"Record count mismatch: {total_reloaded} != {expected_total}"

    # Breakdown by model
    breakdown = reloaded_df.groupBy("model_name").count().collect()
    for row in breakdown:
        print(f"  - Model: {row['model_name']:20} | Count: {row['count']:,} (Expected: {expected_per_model:,})")
        assert row["count"] == expected_per_model, f"Count mismatch for {row['model_name']}"

    # Verify probability bounds [0.0, 1.0]
    invalid_prob_0 = reloaded_df.filter((F.col("probability_no_disease") < 0.0) | (F.col("probability_no_disease") > 1.0)).count()
    invalid_prob_1 = reloaded_df.filter((F.col("probability_disease") < 0.0) | (F.col("probability_disease") > 1.0)).count()
    assert invalid_prob_0 == 0, f"Found {invalid_prob_0} out-of-bounds probability_no_disease!"
    assert invalid_prob_1 == 0, f"Found {invalid_prob_1} out-of-bounds probability_disease!"
    print("[+] Verified probability validity: 100% of probability values fall within [0.0, 1.0].")

    # Verify binary classes
    distinct_actual = sorted([r[0] for r in reloaded_df.select("actual_class").distinct().collect()])
    distinct_pred = sorted([r[0] for r in reloaded_df.select("predicted_class").distinct().collect()])
    print(f"[+] Distinct Actual Classes    : {distinct_actual}")
    print(f"[+] Distinct Predicted Classes : {distinct_pred}")
    assert distinct_actual == [0, 1], f"Unexpected actual classes: {distinct_actual}"
    assert distinct_pred == [0, 1], f"Unexpected predicted classes: {distinct_pred}"

    # Sample predictions
    print("\nSample Test Predictions:")
    reloaded_df.show(10, truncate=False)

    print("[SUCCESS] All prediction validation checks passed perfectly!")
    print("=" * 80)


def main():
    """Main execution flow for Task 9."""
    test_parquet_path = os.path.join("results", "preprocessed", "test")
    models_dir = "models"
    predictions_output_dir = os.path.join("results", "predictions", "test_predictions")

    spark = get_spark_session()
    try:
        print("=" * 80)
        print("PHASE 4 - TASK 9: TEST PREDICTION GENERATION PIPELINE")
        print("=" * 80)

        # 1. Load Test Data
        test_df, test_count = load_test_dataset(spark, test_parquet_path)

        # 2. Load Models
        lr_model, rf_model = load_trained_models(models_dir)

        # 3. Generate Predictions
        lr_preds = generate_predictions_for_model(lr_model, test_df, "Logistic Regression", "features_lr")
        rf_preds = generate_predictions_for_model(rf_model, test_df, "Random Forest", "features_rf")

        # 4. Combine Predictions
        combined_preds = lr_preds.unionByName(rf_preds)

        # 5. Save Predictions
        save_test_predictions(combined_preds, predictions_output_dir)

        # 6. Verify Output
        verify_test_predictions(spark, predictions_output_dir, test_count)

    finally:
        spark.stop()
        print("\nPySpark Evaluation session closed successfully.")


if __name__ == "__main__":
    main()

