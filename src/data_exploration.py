"""
Data Exploration and Quality Analysis Script for Cardiovascular Disease Dataset.

This script performs initial data loading, schema inspection, descriptive statistics,
target distribution analysis, and comprehensive data quality checks (missing values,
duplicates, outliers via IQR, and clinical validity checks) using PySpark.
"""

import os
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType,
    StructField,
    IntegerType,
    FloatType,
    DoubleType
)


def get_spark_session(app_name: str = "CardioDataExploration") -> SparkSession:
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


def load_raw_data(spark: SparkSession, file_path: str):
    """Load raw CSV dataset with semicolon delimiter and inferred schema."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Raw data file not found at: {file_path}")

    df = (
        spark.read
        .option("header", "true")
        .option("sep", ";")
        .option("inferSchema", "true")
        .csv(file_path)
    )
    return df


def inspect_dataset_overview(df):
    """Display dimensions, schema, and column types."""
    row_count = df.count()
    col_count = len(df.columns)

    print("=" * 80)
    print("DATASET OVERVIEW & DIMENSIONS")
    print("=" * 80)
    print(f"Total Records (Rows): {row_count:,}")
    print(f"Total Attributes (Columns): {col_count}")
    print("\nDataset Schema:")
    df.printSchema()

    print("\nColumn Data Types:")
    for field in df.schema.fields:
        print(f"  - {field.name:15}: {field.dataType.simpleString()}")
    print("=" * 80)
    return row_count, col_count


def calculate_descriptive_statistics(df):
    """Calculate and display comprehensive descriptive statistics for all columns."""
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS (PySpark Summary)")
    print("=" * 80)
    
    # We display standard summary statistics
    stats_df = df.summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
    stats_df.show(truncate=False)

    return stats_df


def analyze_target_distribution(df, target_col: str = "cardio"):
    """Analyze target variable distribution and calculate class percentages."""
    print("\n" + "=" * 80)
    print(f"TARGET VARIABLE DISTRIBUTION: '{target_col}'")
    print("=" * 80)

    total_count = df.count()
    dist_df = (
        df.groupBy(target_col)
        .agg(
            F.count("*").alias("count"),
            F.round(F.count("*") * 100.0 / total_count, 4).alias("percentage")
        )
        .orderBy(target_col)
    )
    dist_df.show(truncate=False)

    rows = dist_df.collect()
    for row in rows:
        val = row[target_col]
        label = "Cardiovascular Disease (Positive)" if val == 1 else "No Disease (Negative)"
        print(f"  Class {val} ({label}): {row['count']:,} patients ({row['percentage']:.2f}%)")

    # Ratio
    if len(rows) == 2:
        ratio = rows[0]["count"] / rows[1]["count"]
        print(f"\nClass balance ratio (0 : 1): {ratio:.4f} (Perfect balance: ~1.00)")
    print("=" * 80)


def inspect_categorical_columns(df):
    """Inspect unique values and frequency distributions for categorical variables."""
    print("\n" + "=" * 80)
    print("CATEGORICAL & DISCRETE FEATURES INSPECTION")
    print("=" * 80)

    cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
    total = df.count()

    for col_name in cat_cols:
        print(f"\n--- Distribution for Column: '{col_name}' ---")
        val_df = (
            df.groupBy(col_name)
            .agg(
                F.count("*").alias("count"),
                F.round(F.count("*") * 100.0 / total, 3).alias("pct")
            )
            .orderBy(col_name)
        )
        val_df.show(truncate=False)


def perform_data_quality_analysis(df):
    """
    Perform thorough data quality analysis:
    1. Missing value analysis (nulls & NaNs)
    2. Duplicate analysis (exact with ID, and duplicate records without ID)
    3. Minimum & Maximum value analysis
    4. Outlier detection using IQR
    5. Inconsistent blood pressure detection
    6. Implausible height & weight detection
    7. Validation of categorical values
    """
    print("\n" + "=" * 80)
    print("DATA QUALITY ANALYSIS")
    print("=" * 80)

    total_records = df.count()

    # 1. Missing Values
    print("\n[1] Missing Value Analysis:")
    null_counts = []
    for c in df.columns:
        null_cnt = df.filter(F.col(c).isNull() | F.isnan(F.col(c))).count()
        null_counts.append((c, null_cnt, (null_cnt / total_records) * 100.0))
        print(f"  - {c:15}: {null_cnt:,} missing ({null_cnt/total_records*100:.2f}%)")

    # 2. Duplicate Records Analysis
    print("\n[2] Duplicate Record Analysis:")
    total_with_id = df.count()
    distinct_with_id = df.distinct().count()
    exact_duplicates = total_with_id - distinct_with_id
    print(f"  - Exact duplicates including 'id': {exact_duplicates}")

    feature_cols = [c for c in df.columns if c != "id"]
    distinct_without_id = df.select(feature_cols).distinct().count()
    duplicates_without_id = total_records - distinct_without_id
    print(f"  - Duplicate records excluding 'id': {duplicates_without_id:,}")

    # 3. Min, Max, and Range Analysis
    print("\n[3] Min, Max, and Range Extremes Analysis:")
    for c in ["age", "height", "weight", "ap_hi", "ap_lo"]:
        min_v = df.select(F.min(c)).collect()[0][0]
        max_v = df.select(F.max(c)).collect()[0][0]
        print(f"  - {c:10}: Min = {min_v:>10}, Max = {max_v:>10}")

    # 4. Outlier Detection using IQR
    print("\n[4] Outlier Detection using Interquartile Range (IQR = Q3 - Q1):")
    numerical_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
    iqr_results = {}

    for c in numerical_cols:
        quantiles = df.approxQuantile(c, [0.25, 0.75], 0.001)
        q1, q3 = quantiles[0], quantiles[1]
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = df.filter((F.col(c) < lower_bound) | (F.col(c) > upper_bound)).count()
        pct = (outlier_count / total_records) * 100.0
        iqr_results[c] = {
            "Q1": q1, "Q3": q3, "IQR": iqr,
            "lower_bound": lower_bound, "upper_bound": upper_bound,
            "outlier_count": outlier_count, "outlier_pct": pct
        }
        print(f"  - {c:10}: Q1={q1:>8.2f}, Q3={q3:>8.2f}, IQR={iqr:>8.2f} | "
              f"Bounds=[{lower_bound:>8.2f}, {upper_bound:>8.2f}] | "
              f"Outliers: {outlier_count:>6,} ({pct:.2f}%)")

    # 5. Inconsistent and Implausible Blood Pressure Readings
    print("\n[5] Inconsistent and Implausible Blood Pressure Analysis:")
    neg_hi = df.filter(F.col("ap_hi") <= 0).count()
    neg_lo = df.filter(F.col("ap_lo") <= 0).count()
    inverted_bp = df.filter(F.col("ap_hi") <= F.col("ap_lo")).count()
    extreme_hi = df.filter((F.col("ap_hi") < 60) | (F.col("ap_hi") > 250)).count()
    extreme_lo = df.filter((F.col("ap_lo") < 40) | (F.col("ap_lo") > 150)).count()

    print(f"  - Negative or zero ap_hi (<= 0)       : {neg_hi:,}")
    print(f"  - Negative or zero ap_lo (<= 0)       : {neg_lo:,}")
    print(f"  - Inverted BP (ap_hi <= ap_lo)        : {inverted_bp:,} (Physiologically impossible)")
    print(f"  - Extreme ap_hi (< 60 or > 250 mmHg)  : {extreme_hi:,}")
    print(f"  - Extreme ap_lo (< 40 or > 150 mmHg)  : {extreme_lo:,}")

    # 6. Implausible Height and Weight Values
    print("\n[6] Implausible Anthropometric Measurements (Height & Weight):")
    extreme_height = df.filter((F.col("height") < 120) | (F.col("height") > 220)).count()
    extreme_weight = df.filter((F.col("weight") < 35) | (F.col("weight") > 200)).count()
    print(f"  - Implausible Height (< 120 or > 220 cm): {extreme_height:,}")
    print(f"  - Implausible Weight (< 35 or > 200 kg) : {extreme_weight:,}")

    # 7. Categorical Value Validation
    print("\n[7] Categorical Integrity Verification:")
    valid_specs = {
        "gender": [1, 2],
        "cholesterol": [1, 2, 3],
        "gluc": [1, 2, 3],
        "smoke": [0, 1],
        "alco": [0, 1],
        "active": [0, 1],
        "cardio": [0, 1]
    }
    for c, valid_vals in valid_specs.items():
        invalid_cnt = df.filter(~F.col(c).isin(valid_vals)).count()
        print(f"  - {c:12}: Invalid values count = {invalid_cnt} (Expected set: {valid_vals})")

    # Document Distinction between Invalid Records and Unusual Clinical Observations
    print("\n[8] Clinical Distinction Note:")
    print("  * INVALID RECORDS:")
    print("    - Physiologically impossible entries: ap_hi <= ap_lo (systolic <= diastolic),")
    print("      negative pressures (e.g. -150 mmHg), toddler-like adult weights (< 35 kg),")
    print("      and toddler-like adult heights (< 120 cm). These are data collection/entry errors.")
    print("  * UNUSUAL BUT VALID CLINICAL OBSERVATIONS:")
    print("    - High blood pressures (ap_hi in 160-220 mmHg, ap_lo in 100-140 mmHg) exceed standard")
    print("      IQR upper bounds but represent genuine Stage 2 and Stage 3 Hypertensive patients.")
    print("    - High body weights (weight in 105-180 kg) exceed IQR bounds but represent genuine severe")
    print("      obesity, which is a major recognized cardiovascular risk factor.")
    print("    - These must NOT be discarded as outliers, as doing so would artificially truncate the high-risk cohort.")
    print("=" * 80)

    return {
        "total_records": total_records,
        "exact_duplicates": exact_duplicates,
        "duplicates_without_id": duplicates_without_id,
        "null_counts": null_counts,
        "iqr_results": iqr_results,
        "inverted_bp": inverted_bp,
        "extreme_hi": extreme_hi,
        "extreme_lo": extreme_lo,
        "extreme_height": extreme_height,
        "extreme_weight": extreme_weight
    }


def main():
    """Main execution flow for Task 1 and Task 2."""
    raw_data_path = os.path.join("data", "raw", "cardio_train.csv")
    print(f"Starting PySpark Data Exploration on: {raw_data_path}")

    spark = get_spark_session()
    try:
        # Task 1: Loading and Initial Exploration
        df = load_raw_data(spark, raw_data_path)
        inspect_dataset_overview(df)
        calculate_descriptive_statistics(df)
        analyze_target_distribution(df, target_col="cardio")
        inspect_categorical_columns(df)

        # Task 2: Data Quality Analysis
        perform_data_quality_analysis(df)

    finally:
        spark.stop()
        print("\nPySpark Data Exploration session stopped successfully.")


if __name__ == "__main__":
    main()

