import unittest

import numpy as np

from scripts.utils.prediction_utils import (
    accumulate_price_path,
    accumulate_quantile_price_paths,
)


class PredictionUtilsTests(unittest.TestCase):
    def test_accumulate_quantile_price_paths_mantiene_mediana_compuesta_y_orden(self):
        last_close = 100.0
        median_returns = np.asarray([0.01, 0.02, -0.005, 0.01], dtype=float)
        lower_returns = np.asarray([-0.01, 0.0, -0.02, -0.005], dtype=float)
        upper_returns = np.asarray([0.03, 0.04, 0.01, 0.03], dtype=float)

        median, lower, upper = accumulate_quantile_price_paths(
            last_close,
            median_returns,
            lower_returns,
            upper_returns,
        )
        expected_median = accumulate_price_path(last_close, median_returns)

        np.testing.assert_allclose(median, expected_median, rtol=1e-8, atol=1e-8)
        self.assertTrue(np.isfinite(lower).all())
        self.assertTrue(np.isfinite(median).all())
        self.assertTrue(np.isfinite(upper).all())
        self.assertTrue((lower > 0.0).all())
        self.assertTrue((lower <= median).all())
        self.assertTrue((median <= upper).all())

    def test_accumulate_quantile_price_paths_no_explota_como_cuantiles_pathwise(self):
        last_close = 100.0
        horizon = 20
        median_returns = np.full(horizon, 0.01, dtype=float)
        lower_returns = np.full(horizon, -0.01, dtype=float)
        upper_returns = np.full(horizon, 0.06, dtype=float)

        median, lower, upper = accumulate_quantile_price_paths(
            last_close,
            median_returns,
            lower_returns,
            upper_returns,
        )
        naive_upper = accumulate_price_path(last_close, upper_returns)

        self.assertLess(float(upper[-1]), float(naive_upper[-1]))
        self.assertGreater(float(upper[-1]), float(median[-1]))
        self.assertLess(float(lower[-1]), float(median[-1]))


if __name__ == "__main__":
    unittest.main()
