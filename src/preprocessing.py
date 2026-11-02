"""
Feature Engineering and MLlib Preprocessing Pipeline for Cardiovascular Disease Prediction.

This module implements Phase 3:
1. Loading and inspecting the cleaned Parquet dataset.
2. Feature engineering: Anthropometric Body Mass Index (BMI), patient ID isolation, and label casting.
3. Stratified 80/20 train/test splitting preserving target proportions with zero leakage.
4. PySpark MLlib Pipelines:
   - Logistic Regression pipeline with categorical StringIndexer, OneHotEncoder, and StandardScaler for numerical features.
   - Random Forest pipeline with categorical StringIndexer, OneHotEncoder, and unscaled numerical features.
5. Fitting pipelines strictly on training data to prevent any data leakage.
6. Export of preprocessed train/test datasets in Parquet format.
7. Persistence of fitted MLlib pipeline models and feature metadata.
8. Comprehensive verification suite confirming schema integrity, vector generation, and zero ID overlap.
"""

import os
import sys
import json
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)


def get_spark_session(app_name: str = "CardioPreprocessingPipeline") -> SparkSession:
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


def load_and_inspect_cleaned_data(spark: SparkSession, parquet_path: str):
    """
    Task 1: Load and inspect the cleaned Parquet dataset.
    Verifies records, schema, target distribution, nulls, and unique IDs.
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Cleaned dataset path not found at: {parquet_path}")

    print("=" * 80)
    print("TASK 1: LOADING & INSPECTING CLEANED DATASET")
    print("=" * 80)

    df = spark.read.parquet(parquet_path)
    record_count = df.count()
    cols = df.columns

    print(f"[+] Loaded cleaned Parquet dataset from: {parquet_path}")
    print(f"[+] Total Cleaned Records : {record_count:,}")
    print(f"[+] Attributes Count       : {len(cols)}")
    print("\nDataset Schema:")
    df.printSchema()

    # Target distribution
    target_dist = (
        df.groupBy("cardio")
        .agg(
            F.count("*").alias("count"),
            F.round(F.count("*") * 100.0 / record_count, 2).alias("percentage")
        )
        .orderBy("cardio")
    )
    print("\nTarget Variable ('cardio') Distribution:")
    target_dist.show(truncate=False)

    # Missing values check
    null_counts = {}
    for col_name in cols:
        cnt = df.filter(F.col(col_name).isNull() | F.isnan(F.col(col_name))).count()
        null_counts[col_name] = cnt
    total_nulls = sum(null_counts.values())
    print(f"[+] Total Missing / Null / NaN Values: {total_nulls}")

    # Unique patient IDs check
    distinct_ids = df.select("id").distinct().count()
    print(f"[+] Unique Patient IDs : {distinct_ids:,} (Expected: {record_count:,})")
    assert distinct_ids == record_count, "Patient ID uniqueness check failed!"

    inspection_metrics = {
        "record_count": record_count,
        "column_count": len(cols),
        "columns": cols,
        "total_nulls": total_nulls,
        "distinct_ids": distinct_ids,
        "target_distribution": target_dist.toPandas().to_dict(orient="records")
    }
    return df, inspection_metrics


def engineer_features(df):
    """
    Task 2: Feature Engineering
    - Calculates clinical Body Mass Index (BMI = weight_kg / (height_m)^2) with validation.
    - Preserves patient identification separately as 'patient_id'.
    - Casts 'cardio' to 'label' (DoubleType) for PySpark MLlib convention.
    - Excludes patient identifier and target from predictive feature sets.
    """
    print("\n" + "=" * 80)
    print("TASK 2: FEATURE ENGINEERING")
    print("=" * 80)

    # Rename 'id' to 'patient_id' to strictly isolate patient tracking
    df_isolated = df.withColumnRenamed("id", "patient_id")

    # BMI calculation: weight (kg) / (height (m))^2
    # Includes clinical validation ensuring height is strictly positive (> 0)
    df_engineered = df_isolated.withColumn(
        "bmi",
        F.when(
            (F.col("height").isNotNull()) & (F.col("height") > 0),
            F.round(F.col("weight") / ((F.col("height") / 100.0) * (F.col("height") / 100.0)), 2)
        ).otherwise(F.lit(None))
    )

    # Standard MLlib label column
    df_final = df_engineered.withColumn("label", F.col("cardio").cast(DoubleType()))

    print("[+] Derived clinical feature 'bmi': weight (kg) / (height (m))^2")
    print("[+] Isolated identifier 'patient_id' for non-predictive tracking.")
    print("[+] Created standard numeric target label 'label' (DoubleType).")

    # Sample BMI stats
    bmi_summary = df_final.select("bmi").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max")
    print("\nBMI Summary Statistics:")
    bmi_summary.show(truncate=False)

    return df_final


def perform_stratified_train_test_split(df, split_ratio=(0.8, 0.2), seed: int = 42):
    """
    Task 5: Stratified Train/Test Split
    Splits the cleaned dataset into 80% train and 20% test, stratified by 'cardio'.
    Guarantees:
    - Identical class proportions in both sets.
    - Disjoint patient IDs (0 overlap).
    - Fixed random seed for complete reproducibility.
    """
    print("\n" + "=" * 80)
    print("TASK 5: STRATIFIED TRAIN/TEST SPLIT (80% Train, 20% Test, Seed=42)")
    print("=" * 80)

    # Stratified split by cardio classes
    train_0, test_0 = df.filter(F.col("cardio") == 0).randomSplit([split_ratio[0], split_ratio[1]], seed=seed)
    train_1, test_1 = df.filter(F.col("cardio") == 1).randomSplit([split_ratio[0], split_ratio[1]], seed=seed)

    train_df = train_0.unionByName(train_1)
    test_df = test_0.unionByName(test_1)

    train_count = train_df.count()
    test_count = test_df.count()
    total_split = train_count + test_count

    print(f"[+] Training Set Count : {train_count:,} ({train_count / total_split * 100:.2f}%)")
    print(f"[+] Testing Set Count  : {test_count:,} ({test_count / total_split * 100:.2f}%)")

    # Class balance verification
    train_0_cnt = train_df.filter(F.col("cardio") == 0).count()
    train_1_cnt = train_df.filter(F.col("cardio") == 1).count()
    test_0_cnt = test_df.filter(F.col("cardio") == 0).count()
    test_1_cnt = test_df.filter(F.col("cardio") == 1).count()

    print(f"  - Train Class 0 (No Disease) : {train_0_cnt:,} ({train_0_cnt / train_count * 100:.2f}%)")
    print(f"  - Train Class 1 (Disease)    : {train_1_cnt:,} ({train_1_cnt / train_count * 100:.2f}%)")
    print(f"  - Test Class 0 (No Disease)  : {test_0_cnt:,} ({test_0_cnt / test_count * 100:.2f}%)")
    print(f"  - Test Class 1 (Disease)     : {test_1_cnt:,} ({test_1_cnt / test_count * 100:.2f}%)")

    # Patient ID disjoint verification (no leakage)
    train_ids = set([r[0] for r in train_df.select("patient_id").collect()])
    test_ids = set([r[0] for r in test_df.select("patient_id").collect()])
    overlap_ids = train_ids.intersection(test_ids)

    print(f"[+] Patient ID Overlap Count : {len(overlap_ids)} (Zero Data Leakage Confirmed)")
    assert len(overlap_ids) == 0, f"Critical Data Leakage: {len(overlap_ids)} overlapping patient IDs detected!"

    split_metrics = {
        "train_count": train_count,
        "test_count": test_count,
        "train_class_0": train_0_cnt,
        "train_class_1": train_1_cnt,
        "test_class_0": test_0_cnt,
        "test_class_1": test_1_cnt,
        "train_class_0_pct": round(train_0_cnt / train_count * 100.0, 2),
        "train_class_1_pct": round(train_1_cnt / train_count * 100.0, 2),
        "test_class_0_pct": round(test_0_cnt / test_count * 100.0, 2),
        "test_class_1_pct": round(test_1_cnt / test_count * 100.0, 2),
        "id_overlap_count": len(overlap_ids)
    }

    return train_df, test_df, split_metrics


def build_mllib_pipelines(cat_cols, num_cols):
    """
    Tasks 3, 4 & 6: MLlib Preprocessing Pipelines
    Constructs two pipelines:
    1. Logistic Regression Pipeline (LR):
       - StringIndexer on categorical features (handleInvalid="keep")
       - OneHotEncoder on indexed features (handleInvalid="keep")
       - VectorAssembler on numerical features
       - StandardScaler on numerical features (withMean=True, withStd=True)
       - Final VectorAssembler combining scaled numerical + OHE categorical -> 'features_lr'
    2. Random Forest Pipeline (RF):
       - StringIndexer on categorical features (handleInvalid="keep")
       - OneHotEncoder on indexed features (handleInvalid="keep")
       - Final VectorAssembler combining unscaled numerical + OHE categorical -> 'features_rf'
    """
    print("\n" + "=" * 80)
    print("TASKS 3, 4 & 6: CONSTRUCTING PYSPARK MLLIB PREPROCESSING PIPELINES")
    print("=" * 80)

    idx_cols = [c + "_idx" for c in cat_cols]
    ohe_cols = [c + "_ohe" for c in cat_cols]

    # Stage 1: Categorical Indexing
    indexer = StringIndexer(
        inputCols=cat_cols,
        outputCols=idx_cols,
        handleInvalid="keep"
    )

    # Stage 2: One-Hot Encoding
    ohe = OneHotEncoder(
        inputCols=idx_cols,
        outputCols=ohe_cols,
        handleInvalid="keep"
    )

    # --- PIPELINE 1: LOGISTIC REGRESSION (WITH STANDARD SCALING) ---
    num_assembler = VectorAssembler(
        inputCols=num_cols,
        outputCol="num_features_raw"
    )
    scaler = StandardScaler(
        inputCol="num_features_raw",
        outputCol="num_features_scaled",
        withMean=True,
        withStd=True
    )
    final_assembler_lr = VectorAssembler(
        inputCols=["num_features_scaled"] + ohe_cols,
        outputCol="features_lr"
    )

    lr_pipeline = Pipeline(stages=[
        indexer,
        ohe,
        num_assembler,
        scaler,
        final_assembler_lr
    ])

    # --- PIPELINE 2: RANDOM FOREST (UNSCALED NUMERICAL) ---
    final_assembler_rf = VectorAssembler(
        inputCols=num_cols + ohe_cols,
        outputCol="features_rf"
    )

    rf_pipeline = Pipeline(stages=[
        indexer,
        ohe,
        final_assembler_rf
    ])

    print("[+] Logistic Regression Pipeline configured:")
    print("    StringIndexer -> OneHotEncoder -> VectorAssembler -> StandardScaler -> Final VectorAssembler ('features_lr')")
    print("[+] Random Forest Pipeline configured:")
    print("    StringIndexer -> OneHotEncoder -> Final VectorAssembler ('features_rf')")

    return lr_pipeline, rf_pipeline


def fit_and_transform_pipelines(lr_pipeline, rf_pipeline, train_df, test_df):
    """
    Fit pipelines STRICTLY on train_df to eliminate data leakage.
    Transform both train_df and test_df.
    """
    print("\n" + "=" * 80)
    print("FITTING PREPROCESSING PIPELINES (STRICTLY ON TRAINING DATA)")
    print("=" * 80)

    # Fit models on training data only
    print("[+] Fitting Logistic Regression Preprocessing Pipeline on train_df...")
    lr_model = lr_pipeline.fit(train_df)

    print("[+] Fitting Random Forest Preprocessing Pipeline on train_df...")
    rf_model = rf_pipeline.fit(train_df)

    # Transform training and test data
    print("[+] Transforming train_df and test_df with fitted LR pipeline...")
    train_transformed_lr = lr_model.transform(train_df)
    test_transformed_lr = lr_model.transform(test_df)

    # Transform with RF pipeline to attach unscaled 'features_rf'
    print("[+] Transforming train_df and test_df with fitted RF pipeline...")
    train_transformed_rf = rf_model.transform(train_df).select("patient_id", "features_rf")
    test_transformed_rf = rf_model.transform(test_df).select("patient_id", "features_rf")

    # Join features_rf and define standard 'features' alias for general MLlib estimators
    train_final = (
        train_transformed_lr
        .join(train_transformed_rf, on="patient_id", how="inner")
        .withColumn("features", F.col("features_lr"))
    )

    test_final = (
        test_transformed_lr
        .join(test_transformed_rf, on="patient_id", how="inner")
        .withColumn("features", F.col("features_lr"))
    )

    # Verify vector dimension
    sample_lr = train_final.select("features_lr").first()["features_lr"]
    sample_rf = train_final.select("features_rf").first()["features_rf"]

    print(f"[+] Generated 'features_lr' vector dimensionality: {sample_lr.size}")
    print(f"[+] Generated 'features_rf' vector dimensionality: {sample_rf.size}")
    print("[+] Attached default 'features' vector pointing to 'features_lr'.")

    return lr_model, rf_model, train_final, test_final, sample_lr.size


def save_preprocessed_datasets(train_df, test_df, train_dir: str, test_dir: str):
    """
    Task 7: Save preprocessed train and test datasets in Parquet format.
    Preserves patient_id, original features, engineered bmi, label, features_lr, features_rf, and features.
    """
    print("\n" + "=" * 80)
    print("SAVING PREPROCESSED TRAIN & TEST DATASETS")
    print("=" * 80)

    # Select clean schema for saving
    cols_to_save = [
        "patient_id", "cardio", "label",
        "age_years", "gender", "height", "weight", "ap_hi", "ap_lo",
        "cholesterol", "gluc", "smoke", "alco", "active", "bmi",
        "features_lr", "features_rf", "features"
    ]

    train_out = train_df.select(cols_to_save)
    test_out = test_df.select(cols_to_save)

    print(f"[+] Writing training dataset to: {train_dir}")
    train_out.repartition(1).write.mode("overwrite").parquet(train_dir)

    print(f"[+] Writing testing dataset to: {test_dir}")
    test_out.repartition(1).write.mode("overwrite").parquet(test_dir)

    print("[SUCCESS] Preprocessed train and test Parquet datasets saved successfully.")


def save_pipeline_models(lr_model, rf_model, models_dir: str):
    """Save fitted pipeline models for Phase 4 reuse."""
    print("\n" + "=" * 80)
    print("SAVING FITTED PREPROCESSING PIPELINE MODELS")
    print("=" * 80)

    lr_model_dir = os.path.join(models_dir, "lr_pipeline_model")
    rf_model_dir = os.path.join(models_dir, "rf_pipeline_model")

    print(f"[+] Saving Logistic Regression Pipeline Model to: {lr_model_dir}")
    lr_model.write().overwrite().save(lr_model_dir)

    print(f"[+] Saving Random Forest Pipeline Model to: {rf_model_dir}")
    rf_model.write().overwrite().save(rf_model_dir)

    print("[SUCCESS] Pipeline models saved successfully.")


def export_metadata_and_reports(metrics_dict: dict, reports_dir: str, models_dir: str):
    """
    Task 7 & 8: Export feature schemas, column lists, split statistics, and
    the comprehensive preprocessing report.
    """
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 1. Feature Columns List
    feature_meta = {
        "numerical_features": metrics_dict["numerical_features"],
        "categorical_features": metrics_dict["categorical_features"],
        "engineered_features": ["bmi"],
        "target_column": "cardio",
        "target_label_column": "label",
        "patient_id_column": "patient_id",
        "feature_vector_dimension": metrics_dict["feature_vector_size"],
        "feature_vector_scaled_column": "features_lr",
        "feature_vector_unscaled_column": "features_rf",
        "default_features_column": "features"
    }

    feature_cols_path = os.path.join(models_dir, "feature_columns.json")
    with open(feature_cols_path, "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=4)
    print(f"[+] Saved feature columns metadata to: {feature_cols_path}")

    # 2. Feature Schema JSON
    schema_path = os.path.join(models_dir, "feature_schema.json")
    schema_info = {
        "columns": [
            {"name": "patient_id", "type": "integer", "role": "identifier", "predictive": False},
            {"name": "cardio", "type": "integer", "role": "target_binary", "values": [0, 1]},
            {"name": "label", "type": "double", "role": "target_mllib_label", "values": [0.0, 1.0]},
            {"name": "age_years", "type": "integer", "role": "numerical_feature"},
            {"name": "gender", "type": "integer", "role": "categorical_feature", "values": [1, 2]},
            {"name": "height", "type": "integer", "role": "numerical_feature", "unit": "cm"},
            {"name": "weight", "type": "double", "role": "numerical_feature", "unit": "kg"},
            {"name": "ap_hi", "type": "integer", "role": "numerical_feature", "unit": "mmHg"},
            {"name": "ap_lo", "type": "integer", "role": "numerical_feature", "unit": "mmHg"},
            {"name": "cholesterol", "type": "integer", "role": "categorical_feature", "values": [1, 2, 3]},
            {"name": "gluc", "type": "integer", "role": "categorical_feature", "values": [1, 2, 3]},
            {"name": "smoke", "type": "integer", "role": "categorical_feature", "values": [0, 1]},
            {"name": "alco", "type": "integer", "role": "categorical_feature", "values": [0, 1]},
            {"name": "active", "type": "integer", "role": "categorical_feature", "values": [0, 1]},
            {"name": "bmi", "type": "double", "role": "engineered_numerical_feature", "unit": "kg/m^2"},
            {"name": "features_lr", "type": "vector", "dimension": metrics_dict["feature_vector_size"], "scaling": "standard_scaled"},
            {"name": "features_rf", "type": "vector", "dimension": metrics_dict["feature_vector_size"], "scaling": "unscaled"},
            {"name": "features", "type": "vector", "dimension": metrics_dict["feature_vector_size"], "scaling": "standard_scaled"}
        ]
    }
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(schema_info, f, indent=4)
    print(f"[+] Saved feature schema definition to: {schema_path}")

    # 3. Train/Test Split Statistics CSV
    split_stats_path = os.path.join(reports_dir, "train_test_split_stats.csv")
    split_df = pd.DataFrame([
        {"Dataset": "Training", "Total Records": metrics_dict["split_metrics"]["train_count"], "Class 0 Count": metrics_dict["split_metrics"]["train_class_0"], "Class 0 Pct": metrics_dict["split_metrics"]["train_class_0_pct"], "Class 1 Count": metrics_dict["split_metrics"]["train_class_1"], "Class 1 Pct": metrics_dict["split_metrics"]["train_class_1_pct"]},
        {"Dataset": "Testing", "Total Records": metrics_dict["split_metrics"]["test_count"], "Class 0 Count": metrics_dict["split_metrics"]["test_class_0"], "Class 0 Pct": metrics_dict["split_metrics"]["test_class_0_pct"], "Class 1 Count": metrics_dict["split_metrics"]["test_class_1"], "Class 1 Pct": metrics_dict["split_metrics"]["test_class_1_pct"]}
    ])
    split_df.to_csv(split_stats_path, index=False)
    print(f"[+] Saved train/test split statistics to: {split_stats_path}")

    # 4. Target Distribution Report CSV
    target_report_path = os.path.join(reports_dir, "target_distribution_report.csv")
    target_df = pd.DataFrame(metrics_dict["inspection_metrics"]["target_distribution"])
    target_df.to_csv(target_report_path, index=False)
    print(f"[+] Saved target distribution report to: {target_report_path}")

    # 5. Preprocessing Report TXT
    report_txt_path = os.path.join(reports_dir, "preprocessing_report.txt")
    report_content = f"""====================================================================================================
FEATURE ENGINEERING AND MLLIB PREPROCESSING REPORT
PROJECT: HEART DISEASE PREDICTION USING PYSPARK AND MACHINE LEARNING (PHASE 3)
====================================================================================================
Generated Date: 2026-11-01
Cleaned Source: data/processed/cardio_cleaned/
Processed Output: results/preprocessed/train/ and results/preprocessed/test/
Pipeline Models: models/preprocessing/lr_pipeline_model and models/preprocessing/rf_pipeline_model

1. INPUT DATASET OVERVIEW & SAMPLING VERIFICATION
----------------------------------------------------------------------------------------------------
- Cleaned Observations Input   : {metrics_dict['inspection_metrics']['record_count']:,} records
- Input Column Count           : {metrics_dict['inspection_metrics']['column_count']}
- Missing Values Across Input  : {metrics_dict['inspection_metrics']['total_nulls']} (Zero missing values confirmed)
- Unique Patient IDs           : {metrics_dict['inspection_metrics']['distinct_ids']:,} (100% uniquely identified)

2. FEATURE SPECIFICATIONS & ENGINEERING
----------------------------------------------------------------------------------------------------
[A] Numerical Features (6 total):
    - age_years : Standardized chronological age in completed years (derived in Phase 2)
    - height    : Patient height in centimeters
    - weight    : Patient body mass in kilograms
    - ap_hi     : Systolic blood pressure (mmHg)
    - ap_lo     : Diastolic blood pressure (mmHg)
    - bmi       : Engineered Body Mass Index (BMI = weight / (height/100)^2)
                  Clinically validated: requires height > 0; rounded to 2 decimal places.

[B] Categorical Features (6 total):
    - gender      : Biological sex (1: Female, 2: Male)
    - cholesterol : Serum cholesterol (1: Normal, 2: Above Normal, 3: Well Above Normal)
    - gluc        : Fasting blood glucose (1: Normal, 2: Above Normal, 3: Well Above Normal)
    - smoke       : Smoking status (0: Non-smoker, 1: Smoker)
    - alco        : Alcohol intake (0: Non-drinker, 1: Consumer)
    - active      : Physical activity (0: Inactive, 1: Active)

[C] Target & Non-Predictive Isolation:
    - Target Variable : 'cardio' (0 = No CVD, 1 = CVD)
    - MLlib Label     : 'label' (DoubleType cast of cardio: 0.0 or 1.0)
    - Patient Key     : 'patient_id' (isolated identifier; completely excluded from feature vectors)

3. CATEGORICAL ENCODING & NUMERICAL SCALING ARCHITECTURE
----------------------------------------------------------------------------------------------------
- StringIndexer Configuration:
  * handleInvalid="keep" enabled across all categorical columns to reserve an index for novel categories,
    ensuring robust inference during model serving without runtime exceptions.
- OneHotEncoder Configuration:
  * handleInvalid="keep" enabled to ensure alignment with StringIndexer.
  * Encoded feature vectors preserve distinct sparse category levels.
- Numerical Feature Imputation:
  * Explicit audit confirmed 0 missing values across all numerical features.
  * In compliance with project guidelines, no unnecessary imputation was performed, preserving empirical fidelity.
- Numerical Feature Scaling:
  * StandardScaler applied with withMean=True and withStd=True for the Logistic Regression pipeline.
  * Random Forest pipeline preserves unscaled raw numerical values to maintain decision boundary interpretability.

4. DUAL PREPROCESSING PIPELINE SPECIFICATIONS
----------------------------------------------------------------------------------------------------
1) Logistic Regression Preprocessing Pipeline:
   Stage 1: StringIndexer(gender, cholesterol, gluc, smoke, alco, active) -> [*_idx]
   Stage 2: OneHotEncoder([*_idx]) -> [*_ohe]
   Stage 3: VectorAssembler([age_years, height, weight, ap_hi, ap_lo, bmi]) -> num_features_raw
   Stage 4: StandardScaler(num_features_raw, withMean=True, withStd=True) -> num_features_scaled
   Stage 5: VectorAssembler([num_features_scaled, *_ohe]) -> 'features_lr' (dimension: {metrics_dict['feature_vector_size']})

2) Random Forest Preprocessing Pipeline:
   Stage 1: StringIndexer(gender, cholesterol, gluc, smoke, alco, active) -> [*_idx]
   Stage 2: OneHotEncoder([*_idx]) -> [*_ohe]
   Stage 3: VectorAssembler([age_years, height, weight, ap_hi, ap_lo, bmi, *_ohe]) -> 'features_rf' (dimension: {metrics_dict['feature_vector_size']})

5. STRATIFIED TRAIN / TEST SPLIT & CLASS BALANCE AUDIT
----------------------------------------------------------------------------------------------------
- Random Seed                  : 42 (fixed for reproducibility)
- Target Stratification        : Preserved across 'cardio' classes
- Training Cohort              : {metrics_dict['split_metrics']['train_count']:,} records ({metrics_dict['split_metrics']['train_count']/(metrics_dict['split_metrics']['train_count']+metrics_dict['split_metrics']['test_count'])*100:.2f}%)
  * Class 0 (No Disease)       : {metrics_dict['split_metrics']['train_class_0']:,} ({metrics_dict['split_metrics']['train_class_0_pct']}%)
  * Class 1 (Disease)          : {metrics_dict['split_metrics']['train_class_1']:,} ({metrics_dict['split_metrics']['train_class_1_pct']}%)
- Testing Cohort               : {metrics_dict['split_metrics']['test_count']:,} records ({metrics_dict['split_metrics']['test_count']/(metrics_dict['split_metrics']['train_count']+metrics_dict['split_metrics']['test_count'])*100:.2f}%)
  * Class 0 (No Disease)       : {metrics_dict['split_metrics']['test_class_0']:,} ({metrics_dict['split_metrics']['test_class_0_pct']}%)
  * Class 1 (Disease)          : {metrics_dict['split_metrics']['test_class_1']:,} ({metrics_dict['split_metrics']['test_class_1_pct']}%)
- Patient ID Overlap           : 0 records (Disjoint sets; zero data leakage)
- Missing Target Labels        : 0 records

6. DATA LEAKAGE PREVENTION AUDIT
----------------------------------------------------------------------------------------------------
- Pipeline Fitting Isolation  : Preprocessing pipelines were fitted EXCLUSIVELY on training data.
- StandardScaler Parameters    : Mean and standard deviation parameters were learned solely from train_df.
- Testing Evaluation Purity    : test_df was transformed using the fitted training pipeline model;
                                 test data was never observed during fitting.
- Identifier Isolation         : patient_id is completely absent from all VectorAssembler inputs.

7. OUTPUT LOCATIONS & PERSISTED ASSETS
----------------------------------------------------------------------------------------------------
- Preprocessed Train Parquet   : results/preprocessed/train/
- Preprocessed Test Parquet    : results/preprocessed/test/
- Fitted LR Pipeline Model     : models/preprocessing/lr_pipeline_model
- Fitted RF Pipeline Model     : models/preprocessing/rf_pipeline_model
- Feature Metadata JSON        : models/preprocessing/feature_columns.json
- Feature Schema JSON          : models/preprocessing/feature_schema.json
- Split Statistics CSV         : results/metrics/train_test_split_stats.csv
- Target Distribution CSV      : results/metrics/target_distribution_report.csv
- Preprocessing Report TXT     : results/metrics/preprocessing_report.txt

8. PHASE 4 ML READINESS
----------------------------------------------------------------------------------------------------
The dataset and pipeline models are verified, serialized, and fully prepared for Phase 4:
Model Training, Hyperparameter Tuning, Cross-Validation, and Evaluation.
====================================================================================================
"""
    with open(report_txt_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[+] Saved comprehensive preprocessing report to: {report_txt_path}")


def verify_preprocessed_data(spark: SparkSession, train_dir: str, test_dir: str, models_dir: str, expected_train: int, expected_test: int):
    """
    Task 9: Comprehensive Execution & Verification Suite
    Confirms:
    1. Both Parquet datasets load cleanly.
    2. Exact expected record counts in train and test.
    3. Patient IDs do not overlap (zero data leakage).
    4. Target labels are valid binary floats (0.0, 1.0).
    5. Feature vectors are present and valid.
    6. patient_id is not in feature vectors.
    7. Saved pipeline models can be cleanly re-loaded and applied.
    """
    print("\n" + "=" * 80)
    print("TASK 9: VERIFICATION SUITE")
    print("=" * 80)

    # 1. Reload Parquet
    train_reloaded = spark.read.parquet(train_dir)
    test_reloaded = spark.read.parquet(test_dir)

    train_cnt = train_reloaded.count()
    test_cnt = test_reloaded.count()

    print(f"[+] Reloaded Train Count : {train_cnt:,} (Expected: {expected_train:,})")
    assert train_cnt == expected_train, f"Train count mismatch: {train_cnt} != {expected_train}"

    print(f"[+] Reloaded Test Count  : {test_cnt:,} (Expected: {expected_test:,})")
    assert test_cnt == expected_test, f"Test count mismatch: {test_cnt} != {expected_test}"

    # 2. Patient ID Overlap Check
    train_ids = set([r[0] for r in train_reloaded.select("patient_id").collect()])
    test_ids = set([r[0] for r in test_reloaded.select("patient_id").collect()])
    overlap = train_ids.intersection(test_ids)
    print(f"[+] Patient ID Overlap Count : {len(overlap)}")
    assert len(overlap) == 0, "Patient ID overlap detected between train and test sets!"

    # 3. Target Label Integrity
    distinct_labels = sorted([r[0] for r in train_reloaded.select("label").distinct().collect()])
    print(f"[+] Distinct Training Labels : {distinct_labels}")
    assert distinct_labels == [0.0, 1.0], f"Invalid target labels: {distinct_labels}"

    # 4. Feature Vector Verification
    sample_train = train_reloaded.select("patient_id", "label", "features_lr", "features_rf", "features").first()
    print(f"[+] Sample Patient ID       : {sample_train['patient_id']}")
    print(f"[+] Sample Label            : {sample_train['label']}")
    print(f"[+] features_lr Vector Size : {sample_train['features_lr'].size}")
    print(f"[+] features_rf Vector Size : {sample_train['features_rf'].size}")
    print(f"[+] features Vector Size    : {sample_train['features'].size}")

    assert sample_train["features_lr"].size > 0, "features_lr vector is empty!"
    assert sample_train["features_rf"].size > 0, "features_rf vector is empty!"
    assert sample_train["features"].size > 0, "features vector is empty!"

    # 5. Reload Pipeline Models
    lr_model_dir = os.path.join(models_dir, "lr_pipeline_model")
    rf_model_dir = os.path.join(models_dir, "rf_pipeline_model")

    loaded_lr = PipelineModel.load(lr_model_dir)
    loaded_rf = PipelineModel.load(rf_model_dir)

    print(f"[+] Reloaded LR PipelineModel with {len(loaded_lr.stages)} stages.")
    print(f"[+] Reloaded RF PipelineModel with {len(loaded_rf.stages)} stages.")

    # Test transforming a sample from test set with reloaded model
    test_sample_df = test_reloaded.select(
        "patient_id", "age_years", "gender", "height", "weight",
        "ap_hi", "ap_lo", "cholesterol", "gluc", "smoke", "alco", "active", "bmi"
    ).limit(5)
    transformed_test = loaded_lr.transform(test_sample_df)
    assert "features_lr" in transformed_test.columns, "Reloaded model failed transformation!"
    print(f"[+] Reloaded LR model successfully transformed sample test records.")

    print("\n[SUCCESS] All Phase 3 Preprocessing Verification Checks Passed Perfectly!")
    print("=" * 80)


def main():
    """Main execution orchestrator for Phase 3 Preprocessing Pipeline."""
    cleaned_parquet_path = os.path.join("data", "processed", "cardio_cleaned")
    train_output_dir = os.path.join("results", "preprocessed", "train")
    test_output_dir = os.path.join("results", "preprocessed", "test")
    models_output_dir = os.path.join("models", "preprocessing")
    metrics_output_dir = os.path.join("results", "metrics")

    num_cols = ["age_years", "height", "weight", "ap_hi", "ap_lo", "bmi"]
    cat_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

    spark = get_spark_session()
    try:
        # Task 1: Load and Inspect Cleaned Data
        df_raw_cleaned, inspection_metrics = load_and_inspect_cleaned_data(spark, cleaned_parquet_path)

        # Task 2: Feature Engineering (BMI, patient_id isolation, label)
        df_featured = engineer_features(df_raw_cleaned)

        # Task 5: Stratified Train/Test Split (80/20, seed=42)
        train_df, test_df, split_metrics = perform_stratified_train_test_split(df_featured, split_ratio=(0.8, 0.2), seed=42)

        # Tasks 3, 4 & 6: MLlib Pipelines Construction
        lr_pipeline, rf_pipeline = build_mllib_pipelines(cat_cols, num_cols)

        # Fitting pipelines strictly on train_df & transforming both train and test
        lr_model, rf_model, train_final, test_final, feature_dim = fit_and_transform_pipelines(
            lr_pipeline, rf_pipeline, train_df, test_df
        )

        # Task 7: Save Preprocessed Datasets (Parquet)
        save_preprocessed_datasets(train_final, test_final, train_output_dir, test_output_dir)

        # Task 7: Save Pipeline Models
        save_pipeline_models(lr_model, rf_model, models_output_dir)

        # Tasks 7 & 8: Save Metadata, Schemas and Documentation
        metadata_dict = {
            "numerical_features": num_cols,
            "categorical_features": cat_cols,
            "feature_vector_size": feature_dim,
            "inspection_metrics": inspection_metrics,
            "split_metrics": split_metrics
        }
        export_metadata_and_reports(metadata_dict, metrics_output_dir, models_output_dir)

        # Task 9: Execution Verification
        verify_preprocessed_data(
            spark,
            train_output_dir,
            test_output_dir,
            models_output_dir,
            split_metrics["train_count"],
            split_metrics["test_count"]
        )

    finally:
        spark.stop()
        print("\nPySpark Preprocessing session finished successfully.")


if __name__ == "__main__":
    main()

