import unittest

import pandas as pd

from app.plot_utils import build_benchmark_plot, build_stock_plot


class PlotUtilsTests(unittest.TestCase):
    def test_build_stock_plot_fija_un_rango_y_robusto_ante_bandas_extremas(self):
        config = {"model": {"max_prediction_length": 3}}
        historical_dates = pd.bdate_range("2024-01-02", periods=30)
        ticker_data = pd.DataFrame({"Date": historical_dates, "Ticker": ["AAPL"] * len(historical_dates)})
        original_close = pd.Series([100 + index * 0.5 for index in range(len(historical_dates))])
        median = [116.0, 117.5, 119.0]
        lower = [112.0, 111.0, 109.0]
        upper = [300.0, 700.0, 1200.0]
        forecast_dates = pd.bdate_range(historical_dates[-1] + pd.offsets.BusinessDay(1), periods=3)

        fig, pred_df = build_stock_plot(
            config,
            ticker_data,
            original_close,
            median,
            lower,
            upper,
            "AAPL",
            forecast_dates=forecast_dates,
            historical_period_days=365,
        )

        self.assertIsNotNone(fig.layout.yaxis.range)
        self.assertEqual(len(pred_df), 3)
        self.assertEqual(list(pred_df["Fecha"]), list(forecast_dates))
        self.assertLess(fig.layout.yaxis.range[1], max(upper))
        self.assertGreater(fig.layout.yaxis.range[1], max(median))

    def test_build_benchmark_plot_renderiza_payload_con_split(self):
        payload = {
            "tickers": {
                "AAA": {
                    "split_date": "2024-01-10T00:00:00",
                    "historical_dates": ["2024-01-08T00:00:00", "2024-01-09T00:00:00", "2024-01-10T00:00:00"],
                    "historical_close": [100.0, 101.0, 102.0],
                    "forecast_dates": ["2024-01-11T00:00:00", "2024-01-12T00:00:00"],
                    "actual_close": [103.0, 104.0],
                    "predicted_close": [102.5, 104.5],
                    "metrics": {"MAPE": 0.5},
                }
            }
        }

        fig = build_benchmark_plot(payload, "AAA")

        self.assertEqual(len(fig.data), 3)
        self.assertEqual(fig.layout.title.text, "Benchmark reproducible para AAA")


if __name__ == "__main__":
    unittest.main()
