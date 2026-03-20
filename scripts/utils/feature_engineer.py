from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngineer:
    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def calculate_macd(prices: pd.Series) -> pd.Series:
        exp12 = prices.ewm(span=12, adjust=False).mean()
        exp26 = prices.ewm(span=26, adjust=False).mean()
        return exp12 - exp26

    @staticmethod
    def calculate_roc(prices: pd.Series, period: int = 20) -> pd.Series:
        return 100 * (prices - prices.shift(period)) / prices.shift(period)

    @staticmethod
    def calculate_vwap(group: pd.DataFrame) -> pd.Series:
        typical_price = (group["High"] + group["Low"] + group["Close"]) / 3
        return (typical_price * group["Volume"]).cumsum() / group["Volume"].cumsum()

    def add_features(self, df: pd.DataFrame, sectors_list=None) -> pd.DataFrame:
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], utc=True)

        processed_groups = []
        for _, group in df.groupby("Ticker", group_keys=False):
            group = group.sort_values("Date").copy()
            group["MA10"] = group["Close"].rolling(window=10).mean()
            group["MA50"] = group["Close"].rolling(window=50).mean()
            group["BB_upper"] = group["Close"].rolling(window=20).mean() + 2 * group["Close"].rolling(window=20).std()
            group["Close_to_BB_upper"] = group["Close"] / group["BB_upper"]
            group["RSI"] = self.compute_rsi(group["Close"])
            group["MACD"] = self.calculate_macd(group["Close"])
            group["ROC"] = self.calculate_roc(group["Close"])
            group["VWAP"] = self.calculate_vwap(group)
            group["Momentum_20d"] = group["Close"] - group["Close"].shift(20)
            group["Close_to_MA_ratio"] = group["Close"] / group["MA50"]
            group["Relative_Returns"] = group["Close"].pct_change()
            group["Month"] = group["Date"].dt.month.astype(str)
            group["Day_of_Week"] = group["Date"].dt.dayofweek.astype(str)

            features_to_fill = [
                "MA10",
                "MA50",
                "BB_upper",
                "Close_to_BB_upper",
                "RSI",
                "MACD",
                "ROC",
                "VWAP",
                "Momentum_20d",
                "Close_to_MA_ratio",
            ]
            for feature in features_to_fill:
                if feature in group.columns:
                    group[feature] = group[feature].ffill()
            processed_groups.append(group)

        df = pd.concat(processed_groups, ignore_index=True) if processed_groups else df.iloc[0:0].copy()
        if sectors_list:
            df["Sector"] = pd.Categorical(df["Sector"], categories=sectors_list, ordered=False)
        return df
