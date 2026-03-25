from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer

from scripts.config_manager import ConfigManager
from scripts.utils.artifact_utils import write_checksum, write_metadata
from scripts.utils.data_schema import (
    DAY_OF_WEEK_CATEGORIES,
    KNOWN_CATEGORICAL_FEATURES,
    LOG_FEATURES,
    MONTH_CATEGORIES,
    NUMERIC_FEATURES,
    REQUIRED_NORMALIZER_KEYS,
    REQUIRED_NUMERIC_COLUMNS,
    STATIC_CATEGORICALS,
    TARGET_COLUMN,
    build_artifact_metadata,
    normalize_feature_list,
)
from scripts.utils.feature_engineer import FeatureEngineer
from scripts.utils.repo_layout import resolve_repo_path
from scripts.utils.universe_integrity import build_universe_integrity_report

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["model_name"]
        self.config_manager = ConfigManager(config.get("_meta", {}).get("config_path"))
        self.day_of_week_categories = list(DAY_OF_WEEK_CATEGORIES)
        self.month_categories = list(MONTH_CATEGORIES)
        self.train_processed_df_path = resolve_repo_path(config, config["data"]["train_processed_df_path"])
        self.val_processed_df_path = resolve_repo_path(config, config["data"]["val_processed_df_path"])
        self.processed_data_path = resolve_repo_path(config, config["data"]["processed_data_path"])
        self.gap_days = 10

    def _split_with_gap(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        def split_group(group: pd.DataFrame):
            group = group.sort_values("Date")
            total_days = (group["Date"].max() - group["Date"].min()).days
            train_days = int(0.8 * total_days)
            split_date = group["Date"].min() + pd.Timedelta(days=train_days)
            train = group[group["Date"] <= split_date]
            val_start_date = split_date + pd.Timedelta(days=self.gap_days + 1)
            val = group[group["Date"] >= val_start_date]
            return train, val

        trains = []
        vals = []
        for _, group in df.groupby("Ticker"):
            train_group, val_group = split_group(group)
            if not train_group.empty:
                trains.append(train_group)
            if not val_group.empty:
                vals.append(val_group)

        empty_df = df.iloc[0:0].copy()
        train_df = pd.concat(trains).reset_index(drop=True) if trains else empty_df.copy()
        val_df = pd.concat(vals).reset_index(drop=True) if vals else empty_df.copy()
        return train_df, val_df

    def _drop_invalid_rows(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        if not feature_columns:
            return df.copy()
        filtered_groups = []
        for _, group in df.groupby("Ticker", group_keys=False):
            cleaned_group = group.dropna(subset=feature_columns)
            if not cleaned_group.empty:
                filtered_groups.append(cleaned_group)
        filtered = pd.concat(filtered_groups, ignore_index=True) if filtered_groups else df.iloc[0:0].copy()
        removed_rows = len(df) - len(filtered)
        if removed_rows > 0:
            logger.info("Se han eliminado %s filas de calentamiento o con NaN", removed_rows)
        return filtered

    def _build_validation_context_df(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
        encoder_length = self.config["model"]["max_encoder_length"]
        contextual_groups = []
        for ticker, val_group in val_df.groupby("Ticker", group_keys=False):
            history_group = train_df[train_df["Ticker"] == ticker].sort_values("Date").tail(encoder_length)
            combined = pd.concat([history_group, val_group.sort_values("Date")], ignore_index=True)
            combined = combined.drop_duplicates(subset=["Ticker", "Date"], keep="last")
            if not combined.empty:
                contextual_groups.append(combined)
        if not contextual_groups:
            raise ValueError("No se ha podido construir el contexto de validacion por ticker")
        return pd.concat(contextual_groups, ignore_index=True).sort_values(["Ticker", "Date"]).reset_index(drop=True)

    def _build_normalizer_metadata(self, numeric_features: list[str]) -> dict:
        return build_artifact_metadata(
            self.config,
            numeric_features=numeric_features,
            categorical_features=KNOWN_CATEGORICAL_FEATURES,
            extra={
                "normalizer_keys": normalize_feature_list([*numeric_features, TARGET_COLUMN]),
            },
        )

    def _update_training_run_metadata(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        if "training_run" not in self.config:
            return
        training_run = dict(self.config.get("training_run") or {})
        requested_tickers = [str(ticker) for ticker in training_run.get("requested_tickers", [])]
        downloaded_tickers = [str(ticker) for ticker in training_run.get("downloaded_tickers", requested_tickers)]
        final_tickers_used = sorted(
            dict.fromkeys(
                pd.concat([train_df["Ticker"], val_df["Ticker"]], ignore_index=True).dropna().astype(str).tolist()
            )
        )
        dropped_after_preprocessing = [ticker for ticker in downloaded_tickers if ticker not in final_tickers_used]
        training_run["final_tickers_used"] = final_tickers_used
        training_run["dropped_after_preprocessing"] = dropped_after_preprocessing
        self.config["training_run"] = training_run

    def _assert_training_universe_is_trainable(self) -> None:
        training_run = dict(self.config.get("training_run") or {})
        integrity = dict(training_run.get("universe_integrity") or {})
        if not integrity:
            return
        if integrity.get("training_allowed", True):
            return
        reasons = integrity.get("decision_reasons") or [integrity.get("summary", "Sin detalle")]
        raise ValueError(
            "El universo de entrenamiento no es apto para preprocesado: "
            + " | ".join(str(reason) for reason in reasons if str(reason))
        )

    def _revalidate_training_universe(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> None:
        training_run = dict(self.config.get("training_run") or {})
        if not training_run:
            return

        requested_tickers = list(training_run.get("requested_tickers") or [])
        if not requested_tickers:
            return

        base_integrity = dict(training_run.get("download_universe_integrity") or training_run.get("universe_integrity") or {})
        ticker_integrity = dict(base_integrity.get("ticker_integrity") or {})
        final_df = pd.concat([train_df, val_df], ignore_index=True)
        ticker_payloads: dict[str, dict] = {}
        for ticker in requested_tickers:
            frame = final_df[final_df["Ticker"].astype(str) == str(ticker)].copy()
            previous = dict(ticker_integrity.get(ticker) or {})
            ticker_payloads[ticker] = {
                "frame": frame,
                "source": previous.get("source", "missing"),
                "backend_used": previous.get("backend_used"),
                "errors": list(previous.get("errors", [])),
                "discard_reason": previous.get("discard_reason") or ("descartado_tras_preprocesado" if frame.empty else None),
            }

        report = build_universe_integrity_report(self.config, requested_tickers, ticker_payloads)
        training_run["preprocessed_universe_integrity"] = report.to_dict()
        training_run["universe_integrity"] = report.to_dict()
        training_run["downloaded_tickers"] = list(report.successful_tickers)
        training_run["discarded_tickers"] = list(report.discarded_tickers)
        training_run["discarded_details"] = dict(report.discarded_details)
        self.config["training_run"] = training_run

        if not report.training_allowed:
            reasons = report.decision_reasons or [report.summary]
            raise ValueError(
                "El universo deja de ser apto tras el preprocesado: "
                + " | ".join(str(reason) for reason in reasons if str(reason))
            )
        if report.degraded:
            logger.warning("El universo permanece degradado tras el preprocesado: %s", report.summary)

    def _apply_shared_transformations(self, df: pd.DataFrame, mode: str, ticker: str | None = None) -> tuple[pd.DataFrame, pd.Series | None, list[str]]:
        feature_engineer = FeatureEngineer()
        df = feature_engineer.add_features(df, sectors_list=self.config["model"]["sectors"])
        required_numeric_columns = [feature for feature in REQUIRED_NUMERIC_COLUMNS if feature in df.columns]
        available_numeric_features = [feature for feature in NUMERIC_FEATURES if feature in df.columns]
        df = self._drop_invalid_rows(df, required_numeric_columns)
        if df.empty:
            raise ValueError("No quedan datos validos tras eliminar filas con NaN")

        original_close = None
        if mode == "predict":
            original_close = df["Close"].copy()
            df["Ticker"] = ticker

        df = df.sort_values(["Ticker", "Date"]).copy()
        df["group_id"] = df["Ticker"]
        df["time_idx"] = df.groupby("group_id").cumcount()
        df["Day_of_Week"] = pd.Categorical(df["Date"].dt.dayofweek.astype(str), categories=self.day_of_week_categories, ordered=False)
        df["Month"] = pd.Categorical(df["Date"].dt.month.astype(str), categories=self.month_categories, ordered=False)
        df["Sector"] = pd.Categorical(df["Sector"], categories=self.config["model"]["sectors"], ordered=False)

        for feature in LOG_FEATURES:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        return df, original_close, available_numeric_features

    def process_data(
        self,
        mode: str = "train",
        df: pd.DataFrame | None = None,
        normalizers: dict | None = None,
        ticker: str | None = None,
        historical_mode: bool = False,
        trim_days: int = 0,
    ):
        if mode not in {"train", "predict"}:
            raise ValueError(f"Modo no soportado: {mode}")
        if df is None:
            raise ValueError("DataFrame obligatorio")
        if mode == "predict" and ticker is None:
            raise ValueError("El ticker es obligatorio en modo predict")
        if mode == "train":
            self._assert_training_universe_is_trainable()

        if historical_mode and trim_days > 0 and mode == "predict":
            df = df[df["Date"] >= df["Date"].max() - pd.Timedelta(days=trim_days)]

        df, original_close, numeric_features = self._apply_shared_transformations(df, mode, ticker=ticker)

        if mode == "train":
            train_df, val_df = self._split_with_gap(df)
            if train_df.empty or val_df.empty:
                raise ValueError(f"Split train/val vacio: train={len(train_df)}, val={len(val_df)}")

            self._update_training_run_metadata(train_df, val_df)
            self._revalidate_training_universe(train_df, val_df)
            valid_numeric_features = [feature for feature in numeric_features if feature in train_df.columns]
            expected_metadata = self._build_normalizer_metadata(valid_numeric_features)
            normalizers_path = resolve_repo_path(self.config, self.config_manager.get("paths.normalizers_dir")) / f"{self.model_name}_normalizers.pkl"
            existing_normalizers = {}
            existing_metadata = None

            if normalizers_path.exists():
                existing_normalizers = self.config_manager.load_normalizers(self.model_name)
                existing_metadata = self.config_manager.get_last_normalizers_metadata()

            reuse_existing = bool(
                existing_normalizers
                and existing_metadata
                and existing_metadata.get("schema_hash") == expected_metadata["schema_hash"]
                and existing_metadata.get("config_hash") == expected_metadata["config_hash"]
            )
            if reuse_existing:
                logger.info("Se reutilizan los normalizadores compatibles de %s", self.model_name)
                normalizers = existing_normalizers
            else:
                normalizers = {}

            target_normalizer = normalizers.get(TARGET_COLUMN)
            if target_normalizer is None:
                target_normalizer = TorchNormalizer()
                target_normalizer.fit(train_df[TARGET_COLUMN].values)
                normalizers[TARGET_COLUMN] = target_normalizer

            for feature in list(valid_numeric_features):
                try:
                    if feature not in normalizers:
                        normalizers[feature] = TorchNormalizer()
                        train_df[feature] = normalizers[feature].fit_transform(train_df[feature].values)
                    else:
                        train_df[feature] = normalizers[feature].transform(train_df[feature].values)
                    val_df[feature] = normalizers[feature].transform(val_df[feature].values)
                except Exception as exc:
                    logger.error("Error al normalizar %s: %s", feature, exc)
                    valid_numeric_features.remove(feature)
                    normalizers.pop(feature, None)

            metadata = self._build_normalizer_metadata(valid_numeric_features)
            self.config_manager.save_normalizers(self.model_name, normalizers, metadata=metadata, overwrite=True)

            for cat_col in KNOWN_CATEGORICAL_FEATURES:
                if cat_col in train_df.columns:
                    train_df[cat_col] = train_df[cat_col].astype(str)
                if cat_col in val_df.columns:
                    val_df[cat_col] = val_df[cat_col].astype(str)

            self.train_processed_df_path.parent.mkdir(parents=True, exist_ok=True)
            self.val_processed_df_path.parent.mkdir(parents=True, exist_ok=True)
            train_df.to_parquet(self.train_processed_df_path, index=False)
            val_df.to_parquet(self.val_processed_df_path, index=False)
            val_context_df = self._build_validation_context_df(train_df, val_df)

            categorical_encoders = {
                "Sector": NaNLabelEncoder(add_nan=True),
                "Day_of_Week": NaNLabelEncoder(add_nan=True),
                "Month": NaNLabelEncoder(add_nan=True),
            }
            train_dataset = TimeSeriesDataSet(
                train_df,
                time_idx="time_idx",
                target=TARGET_COLUMN,
                group_ids=["group_id"],
                min_encoder_length=self.config["model"]["min_encoder_length"],
                max_encoder_length=self.config["model"]["max_encoder_length"],
                max_prediction_length=self.config["model"]["max_prediction_length"],
                static_categoricals=STATIC_CATEGORICALS,
                time_varying_known_categoricals=KNOWN_CATEGORICAL_FEATURES,
                time_varying_unknown_reals=valid_numeric_features,
                target_normalizer=target_normalizer,
                allow_missing_timesteps=True,
                add_encoder_length=False,
                categorical_encoders=categorical_encoders,
            )
            val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_context_df, stop_randomization=True, predict=False)

            self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
            train_dataset.save(self.processed_data_path)
            write_metadata(self.processed_data_path, metadata)
            write_checksum(self.processed_data_path)
            logger.info("Columnas procesadas en train_df: %s", train_df.columns.tolist())
            return train_dataset, val_dataset

        if not normalizers:
            raise ValueError("Los normalizadores son obligatorios en modo predict")

        missing_required_normalizers = [feature for feature in REQUIRED_NORMALIZER_KEYS if feature not in normalizers]
        if missing_required_normalizers:
            raise ValueError(
                f"Faltan normalizadores obligatorios para la inferencia: {missing_required_normalizers}"
            )

        for feature in numeric_features:
            if feature in df.columns and feature in normalizers:
                transformed = normalizers[feature].transform(df[feature].values)
                if np.isnan(transformed).any() or np.isinf(transformed).any():
                    raise ValueError(f"La transformacion de {feature} ha generado NaN o Inf")
                df[feature] = transformed

        for cat_col in KNOWN_CATEGORICAL_FEATURES:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str)

        logger.info("Columnas del dataframe procesado: %s", df.columns.tolist())
        return df, original_close
