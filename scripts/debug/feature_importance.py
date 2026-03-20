from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from pytorch_forecasting import TimeSeriesDataSet

from scripts.config_manager import ConfigManager
from scripts.model import build_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class FeatureImportanceAnalyzer:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.config
        self.device = torch.device("cpu")

    def load_model_and_data(self):
        dataset_path = Path(self.config["data"]["processed_data_path"])
        dataset = TimeSeriesDataSet.load(dataset_path)
        model_name = self.config["model_name"]
        checkpoint_path = Path(self.config["paths"]["models_dir"]) / f"{model_name}.pth"
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        hyperparams = checkpoint["hyperparams"]
        if "hidden_continuous_size" not in hyperparams:
            hyperparams["hidden_continuous_size"] = self.config["model"]["hidden_size"] // 2
        model = build_model(dataset, self.config, hyperparams=hyperparams)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        model = model.to(self.device)
        return model, dataset

    def analyze_feature_importance(self, output_csv: str = "data/feature_importance_analysis/feature_importance.csv"):
        model, dataset = self.load_model_and_data()
        dataloader = dataset.to_dataloader(train=False, batch_size=self.config["training"]["batch_size"], num_workers=0)
        x, _ = next(iter(dataloader))
        x = {key: (value.to(self.device) if isinstance(value, torch.Tensor) else value) for key, value in x.items()}

        logger.info("Calculando importancia de features...")
        with torch.no_grad():
            interpretation = model.interpret_output(x)

        importance_rows = []
        for block_name in ("variable_importance", "static_variables", "encoder_variables", "decoder_variables"):
            block = interpretation.get(block_name)
            if isinstance(block, dict):
                for feature_name, importance in block.items():
                    cleaned_name = feature_name.replace("_encoder", "").replace("_decoder", "")
                    importance_rows.append({
                        "Feature": cleaned_name,
                        "Importance": float(importance),
                        "Type": block_name,
                    })

        importance_df = pd.DataFrame(importance_rows)
        if importance_df.empty:
            raise ValueError("No se ha podido extraer importancia de features")
        importance_df = importance_df.groupby(["Feature", "Type"], as_index=False)["Importance"].sum()
        total_sum = importance_df["Importance"].sum()
        importance_df["Importance_Normalized"] = importance_df["Importance"] / total_sum if total_sum > 0 else 0.0
        importance_df = importance_df.sort_values("Importance_Normalized", ascending=False)

        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        importance_df.to_csv(output_path, index=False)
        logger.info("Importancia de features guardada en %s", output_path)
        return importance_df

    def plot_feature_importance(self, importance_df: pd.DataFrame | None = None, output_dir: str = "data/feature_importance_analysis"):
        if importance_df is None:
            importance_df = self.analyze_feature_importance()

        feature_df = importance_df.groupby("Feature", as_index=False)["Importance_Normalized"].sum().sort_values("Importance_Normalized", ascending=False)
        top_df = feature_df.head(10)
        bottom_df = feature_df.tail(10).sort_values("Importance_Normalized")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid")

        for title, data, filename, palette in (
            ("Top 10 features mas importantes", top_df, "top_10_feature_importance.png", "viridis"),
            ("Top 10 features menos importantes", bottom_df, "bottom_10_feature_importance.png", "magma"),
        ):
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x="Importance_Normalized", y="Feature", data=data, palette=palette, hue="Feature", dodge=False, legend=False)
            plt.title(title)
            plt.xlabel("Importancia normalizada")
            plt.ylabel("Feature")
            for index, value in enumerate(data["Importance_Normalized"]):
                ax.text(value + 0.0005, index, f"{value:.4f}", va="center", fontsize=10)
            plot_path = output_path / filename
            plt.tight_layout()
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info("Grafico guardado en %s", plot_path)



def calculate_feature_importance(config_path: str = "config/config.yaml", output_csv: str = "data/feature_importance_analysis/feature_importance.csv"):
    analyzer = FeatureImportanceAnalyzer(config_path)
    importance_df = analyzer.analyze_feature_importance(output_csv)
    analyzer.plot_feature_importance(importance_df)
    return importance_df


if __name__ == "__main__":
    calculate_feature_importance()
