import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer

# Dodaj katalog glowny do sciezek systemowych
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from scripts.runtime_config import ConfigManager
from scripts.utils.data_schema import (
    DAY_OF_WEEK_CATEGORIES,
    KNOWN_CATEGORICAL_FEATURES,
    LOG_FEATURES,
    NUMERIC_FEATURES,
    STATIC_CATEGORICALS,
    TARGET_COLUMN,
    build_artifact_metadata,
)
from scripts.utils.feature_engineer import FeatureEngineer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config['model_name']
        self.config_manager = ConfigManager()
        self.day_of_week_categories = list(DAY_OF_WEEK_CATEGORIES)
        self.train_processed_df_path = Path(config['data']['train_processed_df_path'])
        self.val_processed_df_path = Path(config['data']['val_processed_df_path'])
        self.processed_data_path = Path(config['data']['processed_data_path'])
        self.processed_data_metadata_path = Path(f"{self.processed_data_path}.meta.json")
        self.gap_days = 10

    def _split_with_gap(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        def split_group(group: pd.DataFrame):
            group = group.sort_values('Date')
            total_days = (group['Date'].max() - group['Date'].min()).days
            train_days = int(0.8 * total_days)
            split_date = group['Date'].min() + pd.Timedelta(days=train_days)
            train = group[group['Date'] <= split_date]
            val_start_date = split_date + pd.Timedelta(days=self.gap_days + 1)
            val = group[group['Date'] >= val_start_date]
            return train, val

        trains = []
        vals = []
        for _, group in df.groupby('Ticker'):
            train_group, val_group = split_group(group)
            if not train_group.empty:
                trains.append(train_group)
            if not val_group.empty:
                vals.append(val_group)

        train_df = pd.concat(trains).reset_index(drop=True)
        val_df = pd.concat(vals).reset_index(drop=True)
        return train_df, val_df

    def _drop_invalid_rows(self, df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
        if not feature_columns:
            return df.copy()
        filtered = (
            df.groupby('Ticker', group_keys=False)
            .apply(lambda group: group.dropna(subset=feature_columns))
            .reset_index(drop=True)
        )
        removed_rows = len(df) - len(filtered)
        if removed_rows > 0:
            logger.info(f"Se han eliminado {removed_rows} filas de calentamiento o con NaN.")
        return filtered

    def _build_normalizer_metadata(self, numeric_features: list[str]) -> dict:
        return build_artifact_metadata(
            self.config,
            numeric_features=numeric_features,
            categorical_features=KNOWN_CATEGORICAL_FEATURES,
        )

    def process_data(
        self,
        mode: str = 'train',
        df: pd.DataFrame | None = None,
        normalizers: dict | None = None,
        ticker: str | None = None,
        historical_mode: bool = False,
        trim_days: int = 0,
    ):
        numeric_features = list(NUMERIC_FEATURES)

        if mode == 'train':
            if df is None:
                raise ValueError("DataFrame obligatorio en modo train")
            if historical_mode:
                logger.warning("historical_mode se ignora en modo train")
            if trim_days > 0:
                logger.warning("trim_days se ignora en modo train")
        elif mode == 'predict':
            if df is None or ticker is None:
                raise ValueError("DataFrame y ticker son obligatorios en modo predict")
            if historical_mode and trim_days > 0:
                df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=trim_days)]
            df['Ticker'] = ticker
        else:
            raise ValueError(f"Modo no soportado: {mode}")

        feature_engineer = FeatureEngineer()
        df = feature_engineer.add_features(df, sectors_list=self.config['model']['sectors'])
        available_numeric_features = [feature for feature in numeric_features if feature in df.columns]
        df = self._drop_invalid_rows(df, available_numeric_features)
        if df.empty:
            raise ValueError("No quedan datos validos tras eliminar filas con NaN")

        if mode == 'predict':
            original_close = df['Close'].copy()

        df['group_id'] = ticker if mode == 'predict' else df['Ticker']
        df = df.sort_values(['group_id', 'Date']).copy()
        df['time_idx'] = df.groupby('group_id').cumcount()

        df['Day_of_Week'] = df['Date'].dt.dayofweek.astype(str)
        if df['Day_of_Week'].isna().any():
            df['Day_of_Week'] = df['Day_of_Week'].fillna('0')
        df['Day_of_Week'] = pd.Categorical(df['Day_of_Week'], categories=self.day_of_week_categories, ordered=False)
        df['Sector'] = pd.Categorical(df['Sector'], categories=self.config['model']['sectors'], ordered=False)

        for feature in LOG_FEATURES:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature].clip(lower=0))

        if mode == 'train':
            train_df, val_df = self._split_with_gap(df)
            if train_df.empty or val_df.empty:
                raise ValueError(f"Split train/val vacio: train={len(train_df)}, val={len(val_df)}")

            valid_numeric_features = [feature for feature in numeric_features if feature in train_df.columns]
            metadata = self._build_normalizer_metadata(valid_numeric_features)
            normalizers_path = Path(self.config_manager.get('paths.normalizers_dir')) / f"{self.model_name}_normalizers.pkl"
            existing_normalizers = {}
            existing_metadata = None

            if normalizers_path.exists():
                existing_normalizers = self.config_manager.load_normalizers(self.model_name)
                existing_metadata = self.config_manager.get_last_normalizers_metadata()

            reuse_existing = bool(
                existing_normalizers
                and existing_metadata
                and existing_metadata.get('schema_hash') == metadata['schema_hash']
            )
            if reuse_existing:
                logger.info(f"Se reutilizan los normalizadores compatibles de {self.model_name}.")
                normalizers = existing_normalizers
            else:
                if normalizers_path.exists():
                    logger.warning("Los normalizadores guardados no son compatibles con el esquema actual. Se regeneran.")
                normalizers = {}

            for feature in list(valid_numeric_features):
                try:
                    if feature not in normalizers:
                        normalizers[feature] = TorchNormalizer()
                        train_df[feature] = normalizers[feature].fit_transform(train_df[feature].values)
                    else:
                        train_df[feature] = normalizers[feature].transform(train_df[feature].values)
                    val_df[feature] = normalizers[feature].transform(val_df[feature].values)
                except Exception as e:
                    logger.error(f"Error al normalizar la feature {feature}: {e}")
                    valid_numeric_features.remove(feature)
                    normalizers.pop(feature, None)

            self.config_manager.save_normalizers(
                self.model_name,
                normalizers,
                metadata=metadata,
                overwrite=True,
            )

            for cat_col in KNOWN_CATEGORICAL_FEATURES:
                if cat_col in train_df.columns:
                    train_df[cat_col] = train_df[cat_col].astype(str)
                if cat_col in val_df.columns:
                    val_df[cat_col] = val_df[cat_col].astype(str)

            train_df.to_parquet(self.train_processed_df_path, index=False)
            val_df.to_parquet(self.val_processed_df_path, index=False)

            train_dataset = TimeSeriesDataSet(
                train_df,
                time_idx='time_idx',
                target=TARGET_COLUMN,
                group_ids=['group_id'],
                min_encoder_length=self.config['model']['min_encoder_length'],
                max_encoder_length=self.config['model']['max_encoder_length'],
                max_prediction_length=self.config['model']['max_prediction_length'],
                static_categoricals=STATIC_CATEGORICALS,
                time_varying_known_categoricals=KNOWN_CATEGORICAL_FEATURES,
                time_varying_unknown_reals=valid_numeric_features,
                target_normalizer=normalizers.get(TARGET_COLUMN, TorchNormalizer()),
                allow_missing_timesteps=True,
                add_encoder_length=False,
                categorical_encoders={
                    'Sector': NaNLabelEncoder(add_nan=False),
                    'Day_of_Week': NaNLabelEncoder(add_nan=False),
                    'Month': NaNLabelEncoder(add_nan=False),
                },
            )
            val_dataset = TimeSeriesDataSet.from_dataset(
                train_dataset,
                val_df,
                stop_randomization=True,
                predict=False,
            )

            train_dataset.save(self.processed_data_path)
            with open(self.processed_data_metadata_path, 'w', encoding='utf-8') as metadata_file:
                json.dump(metadata, metadata_file, indent=2, sort_keys=True)
            logger.info(f"Columnas procesadas en train_df: {train_df.columns.tolist()}")
            return train_dataset, val_dataset

        for feature in numeric_features:
            if feature in df.columns and feature in (normalizers or {}):
                transformed = normalizers[feature].transform(df[feature].values)
                if np.isnan(transformed).any() or np.isinf(transformed).any():
                    raise ValueError(f"La transformacion de {feature} ha generado NaN o inf")
                df[feature] = transformed

        for cat_col in KNOWN_CATEGORICAL_FEATURES:
            if cat_col in df.columns:
                df[cat_col] = df[cat_col].astype(str)

        logger.info(f"Columnas del dataframe procesado: {df.columns.tolist()}")
        return df, original_close
