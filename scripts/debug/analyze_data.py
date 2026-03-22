from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scripts.config_manager import ConfigManager
from scripts.data_fetcher import DataFetcher
from scripts.preprocessor import DataPreprocessor
from scripts.utils.data_schema import NUMERIC_FEATURES, TARGET_COLUMN
from scripts.utils.feature_engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/data_analysis.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class DataAnalyzer:
    def __init__(self, config: dict, years: int = 10):
        self.config = config
        self.years = years
        self.config_manager = ConfigManager()
        self.model_name = config["model_name"]
        self.data_fetcher = DataFetcher(self.config_manager, years=years)
        self.data_preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer()
        self.output_dir = Path(self.config["paths"]["logs_dir"]) / "debug"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.normalized_numeric_features = list(NUMERIC_FEATURES)
        self.analysis_features = [*NUMERIC_FEATURES, TARGET_COLUMN]

    def fetch_data(self, tickers: list[str]) -> pd.DataFrame:
        end_date = datetime.now().replace(tzinfo=None)
        start_date = end_date - timedelta(days=self.years * 365)
        df = self.data_fetcher.fetch_many_stocks(tickers, start_date, end_date)
        if not df.empty:
            df["Sector"] = pd.Categorical(df["Sector"], categories=self.config["model"]["sectors"], ordered=False)
            logger.info("Pobrados datos para %s tickers y %s filas", len(df["Ticker"].unique()), len(df))
        return df

    def plot_feature_distribution(self, data: pd.Series, feature: str, normalized: bool = False):
        plt.figure(figsize=(10, 6))
        sns.histplot(data, bins=50, kde=True, color="green" if normalized else "blue")
        plt.title(f"Distribucion de {feature} {'normalizada' if normalized else 'bruta'}")
        plt.xlabel(feature)
        plt.ylabel("Frecuencia")
        output_path = self.output_dir / f"{feature}_{'normalized' if normalized else 'raw'}_all.png"
        plt.savefig(output_path)
        plt.close()
        logger.info("Histograma guardado en %s", output_path)

    def analyze_data(self, df: pd.DataFrame):
        logger.info("Preprocesando datos globales...")
        df_processed = self.feature_engineer.add_features(df, sectors_list=self.config["model"]["sectors"])
        normalizers = self.config_manager.load_normalizers(self.model_name)
        df_normalized = df_processed.copy()

        for feature in self.normalized_numeric_features:
            if feature in df_normalized.columns and feature in normalizers:
                try:
                    df_normalized[feature] = normalizers[feature].transform(df_normalized[feature].values)
                except Exception as exc:
                    logger.error("Error al normalizar %s: %s", feature, exc)

        stats = []
        for feature in self.analysis_features:
            if feature not in df_processed.columns:
                continue
            raw = df_processed[feature].dropna()
            normalized = (
                df_normalized[feature].dropna()
                if feature in self.normalized_numeric_features and feature in df_normalized.columns
                else pd.Series(dtype=float)
            )
            if not raw.empty:
                self.plot_feature_distribution(raw, feature, normalized=False)
            if not normalized.empty:
                self.plot_feature_distribution(normalized, feature, normalized=True)
            stats.append(
                {
                    "Feature": feature,
                    "NaN_count": int(df_processed[feature].isna().sum()),
                    "Zero_count": int((df_processed[feature] == 0).sum()),
                    "Negative_count": int((df_processed[feature] < 0).sum()),
                    "Min": float(raw.min()) if not raw.empty else None,
                    "Max": float(raw.max()) if not raw.empty else None,
                    "Mean": float(raw.mean()) if not raw.empty else None,
                    "Std": float(raw.std()) if not raw.empty else None,
                    "Normalized_Min": float(normalized.min()) if not normalized.empty else None,
                    "Normalized_Max": float(normalized.max()) if not normalized.empty else None,
                    "Normalized_Mean": float(normalized.mean()) if not normalized.empty else None,
                    "Normalized_Std": float(normalized.std()) if not normalized.empty else None,
                }
            )

        stats_df = pd.DataFrame(stats)
        stats_path = self.output_dir / "feature_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info("Estadisticas guardadas en %s", stats_path)

        existing_numeric_features = [feature for feature in self.analysis_features if feature in df_processed.columns]
        numeric_df = df_processed[existing_numeric_features].dropna()
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr()
            corr_output_path = self.output_dir / "correlation_matrix_all.csv"
            correlation_matrix.to_csv(corr_output_path)
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
            plt.title("Matriz de correlacion")
            corr_plot_path = self.output_dir / "correlation_heatmap_all.png"
            plt.savefig(corr_plot_path)
            plt.close()
            logger.info("Heatmap guardado en %s", corr_plot_path)

    def run_analysis(self, tickers: list[str]):
        logger.info("Arrancando analisis para %s tickers", len(tickers))
        df = self.fetch_data(tickers)
        if df.empty:
            logger.error("No se han podido descargar datos")
            return
        self.analyze_data(df)


def main():
    config = ConfigManager().config
    analyzer = DataAnalyzer(config, years=10)
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    analyzer.run_analysis(tickers)


if __name__ == "__main__":
    main()
