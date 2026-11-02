"""Inference with the saved Random Forest model and preprocessing pipeline."""

from __future__ import annotations

import json
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Mapping

from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType, IntegerType, StructField, StructType


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class HeartDiseasePredictor:
    """Load saved inference artifacts once and predict individual patients."""

    def __init__(self, spark: SparkSession) -> None:
        self.spark = spark
        models_dir = PROJECT_ROOT / "models"
        with (models_dir / "preprocessing" / "feature_columns.json").open(
            encoding="utf-8"
        ) as file:
            feature_columns = json.load(file)
        with (models_dir / "preprocessing" / "feature_schema.json").open(
            encoding="utf-8"
        ) as file:
            feature_schema = json.load(file)
        with (models_dir / "best_model" / "model_metadata.json").open(
            encoding="utf-8"
        ) as file:
            model_metadata = json.load(file)

        if model_metadata["selected_best_model"] != "Random Forest":
            raise ValueError("The saved best model is not the expected Random Forest.")
        if model_metadata["features_col"] != feature_columns["feature_vector_unscaled_column"]:
            raise ValueError("Saved model and feature metadata use different feature columns.")

        self.numerical_features = feature_columns["numerical_features"]
        self.categorical_features = feature_columns["categorical_features"]
        self.engineered_features = set(feature_columns["engineered_features"])
        self.feature_vector_column = feature_columns["feature_vector_unscaled_column"]

        schema_by_name = {
            column["name"]: column
            for column in feature_schema["columns"]
        }
        self.feature_types = {
            name: schema_by_name[name]["type"]
            for name in self.numerical_features + self.categorical_features
        }
        self.required_inputs = set(self.numerical_features) - self.engineered_features
        self.required_inputs.update(self.categorical_features)

        self.preprocessing_model = PipelineModel.load(
            str(models_dir / "preprocessing" / "rf_pipeline_model")
        )
        self.model = RandomForestClassificationModel.load(
            str(models_dir / "random_forest")
        )
        indexer_model = next(
            (stage for stage in self.preprocessing_model.stages if hasattr(stage, "labelsArray")),
            None,
        )
        if indexer_model is None:
            raise ValueError("The saved preprocessing pipeline has no fitted category indexer.")
        self.allowed_categories = {
            name: {int(value) for value in labels}
            for name, labels in zip(indexer_model.getInputCols(), indexer_model.labelsArray)
        }
        for name in self.categorical_features:
            if self.allowed_categories.get(name) != set(schema_by_name[name]["values"]):
                raise ValueError(f"Saved category mappings do not match feature metadata for {name}.")
        if self.model.getFeaturesCol() != self.feature_vector_column:
            raise ValueError("The saved model and preprocessing pipeline use different features.")

    def _validate_and_prepare(
        self, patient_features: Mapping[str, object]
    ) -> dict[str, int | float]:
        if not isinstance(patient_features, Mapping):
            raise TypeError("Patient features must be provided as a mapping.")

        supplied = set(patient_features)
        missing = self.required_inputs - supplied
        unexpected = supplied - self.required_inputs
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing required fields: {sorted(missing)}")
            if unexpected:
                details.append(f"unexpected fields: {sorted(unexpected)}")
            raise ValueError("; ".join(details))

        prepared: dict[str, int | float] = {}
        for name in sorted(self.required_inputs):
            value = patient_features[name]
            if isinstance(value, bool):
                raise ValueError(f"{name} must be a valid numeric value, not a boolean.")
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{name} must be a valid numeric value.") from error
            if not math.isfinite(numeric_value):
                raise ValueError(f"{name} must be finite.")

            if self.feature_types[name] == "integer":
                if not numeric_value.is_integer():
                    raise ValueError(f"{name} must be an integer.")
                prepared[name] = int(numeric_value)
            else:
                prepared[name] = numeric_value

            if name in self.allowed_categories and prepared[name] not in self.allowed_categories[name]:
                raise ValueError(
                    f"{name} must be one of {sorted(self.allowed_categories[name])}."
                )

        if not 29 <= prepared["age_years"] <= 64:
            raise ValueError("age_years must be between 29 and 64 years, the observed cleaned-cohort range.")
        if not 120 <= prepared["height"] <= 220:
            raise ValueError("height must be between 120 and 220 cm.")
        if not 35 <= prepared["weight"] <= 200:
            raise ValueError("weight must be between 35 and 200 kg.")
        if not 60 <= prepared["ap_hi"] <= 250:
            raise ValueError("ap_hi must be between 60 and 250 mmHg.")
        if not 40 <= prepared["ap_lo"] <= 150:
            raise ValueError("ap_lo must be between 40 and 150 mmHg.")
        if prepared["ap_hi"] <= prepared["ap_lo"]:
            raise ValueError("ap_hi must be greater than ap_lo.")

        bmi = (
            Decimal(str(prepared["weight"]))
            / (Decimal(str(prepared["height"])) / Decimal("100")) ** 2
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        prepared["bmi"] = float(bmi)
        return prepared

    def predict_patient(
        self, patient_features: Mapping[str, object]
    ) -> dict[str, str | int | float]:
        """Predict from raw clinical features; BMI is derived as in preprocessing."""
        prepared = self._validate_and_prepare(patient_features)
        spark_types = {
            "integer": IntegerType(),
            "double": DoubleType(),
        }
        input_schema = StructType(
            [
                StructField(
                    name,
                    spark_types[self.feature_types[name]],
                    nullable=False,
                )
                for name in self.numerical_features + self.categorical_features
            ]
        )
        input_df = self.spark.createDataFrame([prepared], schema=input_schema)
        transformed = self.preprocessing_model.transform(input_df)
        prediction = (
            self.model.transform(transformed)
            .select(
                "prediction",
                vector_to_array("probability").getItem(0).alias("probability_no_disease"),
                vector_to_array("probability").getItem(1).alias("probability_disease"),
            )
            .first()
        )
        predicted_class = int(prediction["prediction"])
        return {
            "model_name": "Random Forest",
            "predicted_class": predicted_class,
            "probability_no_disease": float(prediction["probability_no_disease"]),
            "probability_disease": float(prediction["probability_disease"]),
            "result": (
                "The model estimated a higher probability of disease."
                if predicted_class == 1
                else "The model estimated a higher probability of no disease."
            ),
            "disclaimer": "This model output is not a medical diagnosis.",
        }
