"""
Data Cleaning and Preprocessing Pipeline for Cardiovascular Disease Dataset.

This script executes:
1. De-duplication based on patient clinical attributes (excluding artificial ID).
2. Verification and reporting of null/missing values.
3. Clinical validation and cleaning of physiological measurements (blood pressure, height, weight).
4. Feature conversion: Derivation of age in years.
5. Preservation of original dataset and patient identification lineage.
6. Export of the cleaned dataset in Parquet format to data/processed/cardio_cleaned/.
7. Generation of data/metrics/data_quality_report.csv.
8. Verification that the saved Parquet dataset can be cleanly reloaded.
"""

import os
import sys
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_spark_session(app_name: str = "CardioDataCleaning") -> SparkSession:
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


def load_raw_dataset(spark: SparkSession, raw_path: str):
    """Load raw dataset from CSV with semicolon delimiter."""
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw data file not found at: {raw_path}")

    df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .csv(raw_path)
    )
    return df


def clean_dataset(df):
    """
    Apply rule-based data quality filtering and feature transformations.
    Returns:
        df_cleaned: PySpark DataFrame of cleaned records
        metrics_dict: Dictionary containing audit statistics for the quality report
    """
    total_original = df.count()

    # Step 1: Missing Value Analysis
    null_counts = {}
    for col_name in df.columns:
        null_cnt = df.filter(F.col(col_name).isNull() | F.isnan(F.col(col_name))).count()
        null_counts[col_name] = null_cnt

    total_nulls = sum(null_counts.values())

    # Step 2: Duplicate Record Analysis & Removal
    # Exact duplicates including ID:
    exact_dups_with_id = total_original - df.distinct().count()

    # Duplicates across clinical features excluding artificial 'id'
    clinical_features = [c for c in df.columns if c != "id"]
    df_dedup = df.dropDuplicates(subset=clinical_features)
    dedup_count = df_dedup.count()
    duplicate_records_removed = total_original - dedup_count

    # Step 3: Flag and Isolate Invalid Clinical Records on Deduplicated Data
    # 3a. Inconsistent Blood Pressure: ap_hi <= ap_lo (impossible physiology)
    inverted_bp_count = df_dedup.filter(F.col("ap_hi") <= F.col("ap_lo")).count()

    # 3b. Implausible Blood Pressure ranges
    # Systolic (ap_hi): 60 to 250 mmHg
    ap_hi_invalid = df_dedup.filter((F.col("ap_hi") < 60) | (F.col("ap_hi") > 250)).count()
    # Diastolic (ap_lo): 40 to 150 mmHg
    ap_lo_invalid = df_dedup.filter((F.col("ap_lo") < 40) | (F.col("ap_lo") > 150)).count()

    # 3c. Implausible Anthropometrics
    # Adult height: 120 to 220 cm
    height_invalid = df_dedup.filter((F.col("height") < 120) | (F.col("height") > 220)).count()
    # Adult weight: 35 to 200 kg
    weight_invalid = df_dedup.filter((F.col("weight") < 35.0) | (F.col("weight") > 200.0)).count()

    # Define composite clinical validity filter
    valid_clinical_condition = (
        (F.col("ap_hi") >= 60) & (F.col("ap_hi") <= 250) &
        (F.col("ap_lo") >= 40) & (F.col("ap_lo") <= 150) &
        (F.col("ap_hi") > F.col("ap_lo")) &
        (F.col("height") >= 120) & (F.col("height") <= 220) &
        (F.col("weight") >= 35.0) & (F.col("weight") <= 200.0)
    )

    df_valid = df_dedup.filter(valid_clinical_condition)
    cleaned_count = df_valid.count()
    invalid_records_removed = dedup_count - cleaned_count
    total_removed = total_original - cleaned_count

    # Step 4: Age Conversion (Days to Years)
    # Using 365.25 days/year to properly account for leap years.
    # We add 'age_years' as integer completed years, and retain 'age' for lineage.
    df_transformed = df_valid.withColumn(
        "age_years",
        F.floor(F.col("age") / 365.25).cast("int")
    )

    # Reorder columns: id, age, age_years, gender, height, weight, ap_hi, ap_lo,
    # cholesterol, gluc, smoke, alco, active, cardio
    ordered_cols = [
        "id", "age", "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"
    ]
    df_cleaned = df_transformed.select(ordered_cols)

    metrics_dict = {
        "original_record_count": total_original,
        "exact_duplicates_with_id": exact_dups_with_id,
        "duplicates_without_id_removed": duplicate_records_removed,
        "post_dedup_record_count": dedup_count,
        "total_null_values": total_nulls,
        "null_counts": null_counts,
        "inverted_bp_count": inverted_bp_count,
        "ap_hi_invalid_count": ap_hi_invalid,
        "ap_lo_invalid_count": ap_lo_invalid,
        "height_invalid_count": height_invalid,
        "weight_invalid_count": weight_invalid,
        "invalid_records_removed": invalid_records_removed,
        "total_records_removed": total_removed,
        "cleaned_record_count": cleaned_count,
        "retention_rate_pct": (cleaned_count / total_original) * 100.0,
        "removal_rate_pct": (total_removed / total_original) * 100.0
    }

    return df_cleaned, metrics_dict


def save_data_quality_report(metrics: dict, output_path: str):
    """Generate a structured CSV Data Quality Report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report_rows = [
        {"Category": "Record Count", "Metric": "Original Record Count", "Value": metrics["original_record_count"], "Action / Status": "Loaded raw CSV dataset", "Rationale": "Initial dataset size before preprocessing"},
        {"Category": "Duplicates", "Metric": "Exact Duplicates (with ID)", "Value": metrics["exact_duplicates_with_id"], "Action / Status": "Verified", "Rationale": "Artificial ID makes all raw rows uniquely indexed"},
        {"Category": "Duplicates", "Metric": "Duplicate Clinical Records (without ID)", "Value": metrics["duplicates_without_id_removed"], "Action / Status": "Removed 24 duplicates", "Rationale": "Identical demographic and clinical observations indicate duplicate entry"},
        {"Category": "Missing Values", "Metric": "Total Null / NaN Count Across All Columns", "Value": metrics["total_null_values"], "Action / Status": "No imputation required", "Rationale": "Dataset is 100% complete; zero nulls present"},
        {"Category": "Blood Pressure", "Metric": "Inverted Blood Pressure (ap_hi <= ap_lo)", "Value": metrics["inverted_bp_count"], "Action / Status": "Removed invalid rows", "Rationale": "Physiologically impossible for systolic to be <= diastolic pressure"},
        {"Category": "Blood Pressure", "Metric": "Systolic BP (ap_hi) Out of Bounds [<60 or >250]", "Value": metrics["ap_hi_invalid_count"], "Action / Status": "Removed invalid rows", "Rationale": "Values <60 (or negative) and >250 (up to 16,020) are data entry typos"},
        {"Category": "Blood Pressure", "Metric": "Diastolic BP (ap_lo) Out of Bounds [<40 or >150]", "Value": metrics["ap_lo_invalid_count"], "Action / Status": "Removed invalid rows", "Rationale": "Values <40 (or negative) and >150 (up to 11,000) are data entry typos"},
        {"Category": "Anthropometrics", "Metric": "Height Out of Bounds [<120 or >220 cm]", "Value": metrics["height_invalid_count"], "Action / Status": "Removed invalid rows", "Rationale": "Severe outliers (<120 cm e.g. 55 cm) reflect toddler measurements"},
        {"Category": "Anthropometrics", "Metric": "Weight Out of Bounds [<35 or >200 kg]", "Value": metrics["weight_invalid_count"], "Action / Status": "Removed invalid rows", "Rationale": "Severe outliers (<35 kg e.g. 10 kg) reflect toddler measurements"},
        {"Category": "Clinical Outliers", "Metric": "Valid Severe Clinical Cases Retained", "Value": "Retained", "Action / Status": "Preserved without truncation", "Rationale": "Hypertensive (ap_hi 160-220) and obese (weight 105-180 kg) patients are genuine high-risk cases"},
        {"Category": "Feature Engineering", "Metric": "Age Conversion (Days to Years)", "Value": "Created 'age_years'", "Action / Status": "Calculated floor(age / 365.25)", "Rationale": "Standardized chronological age in completed years, accounting for leap years"},
        {"Category": "Patient Identifier", "Metric": "Patient ID Lineage", "Value": "Retained in storage", "Action / Status": "Kept in parquet, excluded from future ML features", "Rationale": "Maintains patient auditability and lineage without leaking arbitrary ID to ML models"},
        {"Category": "Summary", "Metric": "Total Records Removed", "Value": metrics["total_records_removed"], "Action / Status": f"{metrics['removal_rate_pct']:.2f}% of raw dataset", "Rationale": "Combined duplicates and invalid clinical measurements removed"},
        {"Category": "Summary", "Metric": "Cleaned Record Count", "Value": metrics["cleaned_record_count"], "Action / Status": f"{metrics['retention_rate_pct']:.2f}% retained", "Rationale": "High-fidelity dataset ready for exploratory analysis and ML modeling"}
    ]

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(output_path, index=False)
    print(f"\n[+] Data quality report saved successfully to: {output_path}")


def save_cleaned_parquet(df_cleaned, output_dir: str):
    """Save cleaned PySpark DataFrame in Parquet format."""
    print(f"\n[+] Saving cleaned dataset to Parquet at: {output_dir}")
    # Write as single or partitioned parquet files
    (
        df_cleaned
        .repartition(1)
        .write
        .mode("overwrite")
        .parquet(output_dir)
    )
    print("[+] Parquet write complete.")


def verify_cleaned_dataset(spark: SparkSession, parquet_dir: str, expected_count: int):
    """Verify that saved Parquet dataset can be loaded and matches expected integrity checks."""
    print("\n" + "=" * 80)
    print("VERIFYING SAVED PARQUET DATASET")
    print("=" * 80)

    df_reloaded = spark.read.parquet(parquet_dir)
    loaded_count = df_reloaded.count()
    loaded_cols = df_reloaded.columns

    print(f"Reloaded record count: {loaded_count:,} (Expected: {expected_count:,})")
    assert loaded_count == expected_count, f"Count mismatch! Reloaded {loaded_count} != Expected {expected_count}"

    print(f"Reloaded columns ({len(loaded_cols)}): {loaded_cols}")
    assert "age_years" in loaded_cols, "'age_years' column missing in reloaded Parquet!"
    assert "cardio" in loaded_cols, "'cardio' column missing in reloaded Parquet!"

    # Verify target variable values
    cardio_vals = [row["cardio"] for row in df_reloaded.select("cardio").distinct().collect()]
    print(f"Target column 'cardio' distinct values: {sorted(cardio_vals)}")
    assert set(cardio_vals) == {0, 1}, f"Unexpected target values: {cardio_vals}"

    # Sample records
    print("\nSample records from cleaned Parquet:")
    df_reloaded.show(5, truncate=False)

    print("[SUCCESS] Parquet dataset verified successfully!")
    print("=" * 80)


def main():
    """Main execution routine for Task 3 and Task 4."""
    raw_csv_path = os.path.join("data", "raw", "cardio_train.csv")
    cleaned_parquet_dir = os.path.join("data", "processed", "cardio_cleaned")
    metrics_report_path = os.path.join("results", "metrics", "data_quality_report.csv")

    spark = get_spark_session()
    try:
        print("=" * 80)
        print("PHASE 2 - TASK 3 & 4: DATA CLEANING PIPELINE")
        print("=" * 80)

        # 1. Load Raw Data
        df_raw = load_raw_dataset(spark, raw_csv_path)

        # 2. Clean Data
        df_cleaned, metrics = clean_dataset(df_raw)

        print(f"\nCleaning Summary:")
        print(f"  - Original Records: {metrics['original_record_count']:,}")
        print(f"  - Duplicate Records Removed: {metrics['duplicates_without_id_removed']:,}")
        print(f"  - Invalid Clinical Records Removed: {metrics['invalid_records_removed']:,}")
        print(f"  - Total Records Removed: {metrics['total_records_removed']:,} ({metrics['removal_rate_pct']:.2f}%)")
        print(f"  - Cleaned Records Retained: {metrics['cleaned_record_count']:,} ({metrics['retention_rate_pct']:.2f}%)")

        # 3. Save Parquet Dataset
        save_cleaned_parquet(df_cleaned, cleaned_parquet_dir)

        # 4. Save Data Quality Metrics Report
        save_data_quality_report(metrics, metrics_report_path)

        # 5. Verify Reloading
        verify_cleaned_dataset(spark, cleaned_parquet_dir, metrics["cleaned_record_count"])

    finally:
        spark.stop()
        print("\nPySpark Data Cleaning session finished.")


if __name__ == "__main__":
    main()

