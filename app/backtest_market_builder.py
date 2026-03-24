from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from again_econ.contracts import MarketBar, MarketFrame
from scripts.data_fetcher import DataFetcher, FreshDataRequiredError
from scripts.runtime_config import ConfigManager
from scripts.utils.universe_integrity import UniverseIntegrityReport


@dataclass(frozen=True)
class MarketFrameBuildResult:
    market_frame: MarketFrame
    frame: pd.DataFrame
    requested_tickers: tuple[str, ...]
    effective_tickers: tuple[str, ...]
    discarded_tickers: tuple[str, ...]
    integrity_report: UniverseIntegrityReport
    source_summary: dict[str, int]
    provenance_by_ticker: dict[str, dict[str, Any]]
    warnings: tuple[str, ...] = ()


class MarketFrameBuilder:
    REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector")

    def __init__(self, fetcher: DataFetcher, *, max_gap_days: int = 10) -> None:
        self._fetcher = fetcher
        self._max_gap_days = int(max_gap_days)

    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager, years: int, *, max_gap_days: int = 10) -> "MarketFrameBuilder":
        return cls(DataFetcher(config_manager, years), max_gap_days=max_gap_days)

    def build(
        self,
        tickers: list[str] | tuple[str, ...],
        start_date: datetime,
        end_date: datetime,
        *,
        allow_local_fallback: bool,
    ) -> MarketFrameBuildResult:
        requested_tickers = tuple(self._normalize_tickers(tickers))
        if not requested_tickers:
            raise ValueError("El backtesting requiere al menos un ticker valido")

        combined, integrity_report = self._fetcher.fetch_many_stocks_with_report(
            list(requested_tickers),
            start_date,
            end_date,
        )
        if integrity_report.fallback_tickers and not allow_local_fallback:
            fallback = ", ".join(integrity_report.fallback_tickers)
            raise FreshDataRequiredError(
                f"El modo actual no permite fallback a cache local. Tickers afectados: {fallback}"
            )
        if combined.empty:
            raise ValueError("No se pudieron construir datos de mercado para el universo solicitado")

        normalized = self._normalize_frame(combined)
        warnings = self._detect_critical_gaps(normalized)
        market_frame = self._build_market_frame(normalized)
        provenance_by_ticker = self._build_provenance_map(integrity_report)
        source_summary = self._build_source_summary(provenance_by_ticker, integrity_report.successful_tickers)
        return MarketFrameBuildResult(
            market_frame=market_frame,
            frame=normalized,
            requested_tickers=requested_tickers,
            effective_tickers=tuple(integrity_report.successful_tickers),
            discarded_tickers=tuple(integrity_report.discarded_tickers),
            integrity_report=integrity_report,
            source_summary=source_summary,
            provenance_by_ticker=provenance_by_ticker,
            warnings=warnings,
        )

    @staticmethod
    def _normalize_tickers(tickers: list[str] | tuple[str, ...]) -> list[str]:
        normalized = []
        for ticker in tickers:
            value = str(ticker).strip().upper()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        normalized = frame.copy()
        missing = [column for column in self.REQUIRED_COLUMNS if column not in normalized.columns]
        if missing:
            raise ValueError(f"El dataframe de mercado no contiene columnas obligatorias: {missing}")

        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.tz_localize(None)
        normalized["Ticker"] = normalized["Ticker"].astype(str).str.strip().str.upper()
        normalized["Sector"] = normalized["Sector"].astype(str).str.strip().replace({"": "Unknown"})
        for column in ("Open", "High", "Low", "Close", "Volume"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        normalized = normalized.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]).copy()
        if normalized.empty:
            raise ValueError("El dataframe de mercado queda vacio tras normalizar columnas y tipos")

        duplicates = normalized[normalized.duplicated(subset=["Ticker", "Date"], keep=False)]
        if not duplicates.empty:
            examples = duplicates[["Ticker", "Date"]].astype(str).drop_duplicates().head(5).to_dict(orient="records")
            raise ValueError(f"Se detectaron barras duplicadas por ticker/fecha en el mercado: {examples}")

        normalized = normalized.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        return normalized[list(self.REQUIRED_COLUMNS)]

    def _detect_critical_gaps(self, frame: pd.DataFrame) -> tuple[str, ...]:
        warnings: list[str] = []
        for ticker, ticker_frame in frame.groupby("Ticker", sort=True):
            ordered_dates = pd.to_datetime(ticker_frame["Date"]).sort_values().reset_index(drop=True)
            if len(ordered_dates) < 2:
                continue
            gaps = ordered_dates.diff().dt.days.fillna(0).astype(int)
            critical = gaps[gaps > self._max_gap_days]
            if critical.empty:
                continue
            idx = int(critical.index[0])
            warnings.append(
                (
                    f"{ticker} presenta un hueco temporal de {int(critical.iloc[0])} dias entre "
                    f"{ordered_dates.iloc[idx - 1].date()} y {ordered_dates.iloc[idx].date()}"
                )
            )
        return tuple(warnings)

    @staticmethod
    def _build_market_frame(frame: pd.DataFrame) -> MarketFrame:
        bars = tuple(
            MarketBar(
                instrument_id=str(row.Ticker),
                timestamp=pd.Timestamp(row.Date).to_pydatetime(),
                open=float(row.Open),
                high=float(row.High),
                low=float(row.Low),
                close=float(row.Close),
                volume=float(row.Volume),
            )
            for row in frame.itertuples(index=False)
        )
        return MarketFrame(bars=bars)

    @staticmethod
    def _build_provenance_map(report: UniverseIntegrityReport) -> dict[str, dict[str, Any]]:
        provenance: dict[str, dict[str, Any]] = {}
        for ticker, entry in report.ticker_integrity.items():
            provenance[ticker] = {
                "source": entry.source,
                "used_fallback": entry.used_fallback,
                "freshness": entry.freshness,
                "rows_obtained": entry.rows_obtained,
                "backend_used": entry.backend_used,
                "errors": list(entry.errors),
                "discard_reason": entry.discard_reason,
            }
        return provenance

    @staticmethod
    def _build_source_summary(
        provenance_by_ticker: dict[str, dict[str, Any]],
        successful_tickers: list[str],
    ) -> dict[str, int]:
        counts: dict[str, int] = {}
        for ticker in successful_tickers:
            source = str((provenance_by_ticker.get(ticker) or {}).get("source") or "unknown")
            counts[source] = counts.get(source, 0) + 1
        return counts
