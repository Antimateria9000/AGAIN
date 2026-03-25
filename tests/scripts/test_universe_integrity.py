import unittest

import pandas as pd

from scripts.utils.universe_integrity import build_universe_integrity_report


def _frame(ticker: str, start: str, periods: int) -> pd.DataFrame:
    dates = pd.date_range(start=start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [10.0 + idx for idx in range(periods)],
            "High": [10.5 + idx for idx in range(periods)],
            "Low": [9.5 + idx for idx in range(periods)],
            "Close": [10.2 + idx for idx in range(periods)],
            "Volume": [1000 + idx for idx in range(periods)],
            "Ticker": [ticker] * periods,
            "Sector": ["Financials"] * periods,
        }
    )


class UniverseIntegrityTests(unittest.TestCase):
    def _config(self, **training_universe_overrides) -> dict:
        training_universe = {
            "minimum_group_tickers": 2,
            "minimum_group_tickers_abs": 2,
            "minimum_group_coverage_ratio": 0.75,
            "minimum_rows_per_ticker": 5,
            "minimum_common_overlap_days": 3,
            "abort_if_anchor_missing": True,
            "abort_if_too_many_fallback_tickers": True,
            "abort_on_cached_backfill": False,
            "allow_degraded_universe": False,
            "maximum_fallback_tickers": 0,
            "maximum_fallback_ratio": 0.0,
            **training_universe_overrides,
        }
        return {
            "model": {
                "max_encoder_length": 10,
                "max_prediction_length": 5,
                "min_encoder_length": 5,
            },
            "training_universe": training_universe,
            "training_run": {
                "mode": "predefined_group",
                "anchor_ticker": training_universe.get("anchor_ticker"),
            },
        }

    def test_anchor_missing_aborts(self):
        config = self._config(anchor_ticker="BBVA.MC")
        report = build_universe_integrity_report(
            config,
            ["BBVA.MC", "SAN.MC"],
            {
                "SAN.MC": {
                    "frame": _frame("SAN.MC", "2024-01-01", 8),
                    "source": "fresh_network",
                }
            },
        )

        self.assertEqual(report.decision, "ABORT")
        self.assertFalse(report.training_allowed)
        self.assertFalse(report.anchor_usable)

    def test_coverage_below_threshold_aborts(self):
        config = self._config(minimum_group_coverage_ratio=0.8)
        report = build_universe_integrity_report(
            config,
            ["A", "B", "C", "D"],
            {
                "A": {"frame": _frame("A", "2024-01-01", 8), "source": "fresh_network"},
                "B": {"frame": _frame("B", "2024-01-01", 8), "source": "fresh_network"},
            },
        )

        self.assertEqual(report.decision, "ABORT")
        self.assertLess(report.coverage_ratio, 0.8)

    def test_rows_insufficient_per_ticker_abort(self):
        config = self._config(minimum_rows_per_ticker=6)
        report = build_universe_integrity_report(
            config,
            ["AAPL", "MSFT"],
            {
                "AAPL": {"frame": _frame("AAPL", "2024-01-01", 5), "source": "fresh_network"},
                "MSFT": {"frame": _frame("MSFT", "2024-01-01", 8), "source": "fresh_network"},
            },
        )

        self.assertEqual(report.decision, "ABORT")
        self.assertIn("AAPL", report.discarded_tickers)
        self.assertEqual(report.ticker_integrity["AAPL"].discard_reason, "filas_insuficientes<6>")

    def test_common_overlap_insufficient_aborts(self):
        config = self._config(minimum_group_coverage_ratio=1.0, minimum_common_overlap_days=3)
        report = build_universe_integrity_report(
            config,
            ["AAPL", "MSFT"],
            {
                "AAPL": {"frame": _frame("AAPL", "2024-01-01", 8), "source": "fresh_network"},
                "MSFT": {"frame": _frame("MSFT", "2024-02-01", 8), "source": "fresh_network"},
            },
        )

        self.assertEqual(report.decision, "ABORT")
        self.assertEqual(report.common_overlap_days, 0)

    def test_too_many_fallback_tickers_abort(self):
        config = self._config(maximum_fallback_tickers=0, maximum_fallback_ratio=0.0)
        report = build_universe_integrity_report(
            config,
            ["AAPL", "MSFT"],
            {
                "AAPL": {"frame": _frame("AAPL", "2024-01-01", 8), "source": "fresh_network"},
                "MSFT": {"frame": _frame("MSFT", "2024-01-01", 8), "source": "local_cache"},
            },
        )

        self.assertEqual(report.decision, "ABORT")
        self.assertEqual(report.fallback_tickers, ["MSFT"])

    def test_degraded_allowed_is_marked_but_not_promoted(self):
        config = self._config(
            allow_degraded_universe=True,
            abort_if_too_many_fallback_tickers=False,
            maximum_fallback_tickers=1,
            maximum_fallback_ratio=1.0,
            minimum_group_coverage_ratio=1.0,
        )
        report = build_universe_integrity_report(
            config,
            ["AAPL", "MSFT"],
            {
                "AAPL": {"frame": _frame("AAPL", "2024-01-01", 8), "source": "fresh_network"},
                "MSFT": {"frame": _frame("MSFT", "2024-01-01", 8), "source": "local_cache"},
            },
        )

        self.assertEqual(report.decision, "DEGRADED_ALLOWED")
        self.assertTrue(report.training_allowed)
        self.assertFalse(report.can_promote_canonical)

    def test_valid_clean_universe_continues(self):
        config = self._config(anchor_ticker="AAPL")
        report = build_universe_integrity_report(
            config,
            ["AAPL", "MSFT"],
            {
                "AAPL": {"frame": _frame("AAPL", "2024-01-01", 8), "source": "fresh_network"},
                "MSFT": {"frame": _frame("MSFT", "2024-01-01", 8), "source": "fresh_network"},
            },
        )

        self.assertEqual(report.decision, "CONTINUE_CLEAN")
        self.assertTrue(report.training_allowed)
        self.assertTrue(report.can_promote_canonical)
        self.assertEqual(report.fallback_tickers, [])


if __name__ == "__main__":
    unittest.main()
