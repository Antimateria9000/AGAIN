from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
import math
from typing import Any

import pandas as pd

from again_econ.contracts import MarketBar, MarketFrame
from scripts.data_fetcher import DataFetcher, FreshDataRequiredError
from scripts.runtime_config import ConfigManager
from scripts.utils.universe_integrity import UniverseIntegrityReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarketDataQualityReport:
    input_rows: int
    output_rows: int
    repaired_rows: int = 0
    dropped_rows: int = 0
    duplicate_rows_resolved: int = 0
    repaired_reason_counts: dict[str, int] = field(default_factory=dict)
    dropped_reason_counts: dict[str, int] = field(default_factory=dict)
    repaired_examples: tuple[dict[str, Any], ...] = ()
    dropped_examples: tuple[dict[str, Any], ...] = ()


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
    quality_report: MarketDataQualityReport
    warnings: tuple[str, ...] = ()


class MarketFrameBuilder:
    REQUIRED_COLUMNS = ("Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector")
    MAX_EXAMPLES = 10

    def __init__(
        self,
        fetcher: DataFetcher,
        *,
        max_gap_days: int = 10,
        ohlc_repair_abs_tol: float = 1e-9,
        ohlc_repair_rel_tol: float = 1e-7,
    ) -> None:
        self._fetcher = fetcher
        self._max_gap_days = int(max_gap_days)
        self._ohlc_repair_abs_tol = float(ohlc_repair_abs_tol)
        self._ohlc_repair_rel_tol = float(ohlc_repair_rel_tol)

    @classmethod
    def from_config_manager(
        cls,
        config_manager: ConfigManager,
        years: int,
        *,
        max_gap_days: int = 10,
        ohlc_repair_abs_tol: float = 1e-9,
        ohlc_repair_rel_tol: float = 1e-7,
    ) -> "MarketFrameBuilder":
        return cls(
            DataFetcher(config_manager, years),
            max_gap_days=max_gap_days,
            ohlc_repair_abs_tol=ohlc_repair_abs_tol,
            ohlc_repair_rel_tol=ohlc_repair_rel_tol,
        )

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

        normalized, quality_report, quality_warnings = self._normalize_frame(combined)
        warnings = list(quality_warnings)
        warnings.extend(self._detect_critical_gaps(normalized))
        market_frame = self._build_market_frame(normalized)
        provenance_by_ticker = self._build_provenance_map(integrity_report)
        effective_tickers = tuple(sorted(normalized["Ticker"].astype(str).unique()))
        requested_set = set(requested_tickers)
        discarded_set = set(integrity_report.discarded_tickers)
        discarded_set.update(sorted(requested_set.difference(effective_tickers)))
        for ticker in sorted(discarded_set):
            if ticker not in provenance_by_ticker:
                provenance_by_ticker[ticker] = {}
            if ticker not in effective_tickers:
                provenance_by_ticker[ticker]["builder_filtered_out"] = True
                provenance_by_ticker[ticker].setdefault("discard_reason", "market_builder_filtered_all_rows")
        source_summary = self._build_source_summary(provenance_by_ticker, list(effective_tickers))
        for warning in dict.fromkeys(warnings):
            logger.warning("Backtesting market builder: %s", warning)
        return MarketFrameBuildResult(
            market_frame=market_frame,
            frame=normalized,
            requested_tickers=requested_tickers,
            effective_tickers=effective_tickers,
            discarded_tickers=tuple(sorted(discarded_set)),
            integrity_report=integrity_report,
            source_summary=source_summary,
            provenance_by_ticker=provenance_by_ticker,
            quality_report=quality_report,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _normalize_tickers(tickers: list[str] | tuple[str, ...]) -> list[str]:
        normalized = []
        for ticker in tickers:
            value = str(ticker).strip().upper()
            if value and value not in normalized:
                normalized.append(value)
        return normalized

    def _normalize_frame(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, MarketDataQualityReport, tuple[str, ...]]:
        normalized = frame.copy()
        input_rows = int(len(normalized))
        warnings: list[str] = []
        missing = [column for column in self.REQUIRED_COLUMNS if column not in normalized.columns]
        if missing:
            raise ValueError(f"El dataframe de mercado no contiene columnas obligatorias: {missing}")

        normalized["Date"] = pd.to_datetime(normalized["Date"], errors="coerce").dt.tz_localize(None)
        normalized["Ticker"] = normalized["Ticker"].astype(str).str.strip().str.upper()
        normalized["Sector"] = (
            normalized["Sector"]
            .where(normalized["Sector"].notna(), "Unknown")
            .astype(str)
            .str.strip()
            .replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})
        )
        for column in ("Open", "High", "Low", "Close", "Volume"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

        rows_before_dropna = len(normalized)
        normalized = normalized.dropna(subset=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]).copy()
        missing_required_rows = int(rows_before_dropna - len(normalized))
        dropped_reason_counts: dict[str, int] = {}
        repaired_reason_counts: dict[str, int] = {}
        repaired_examples: list[dict[str, Any]] = []
        dropped_examples: list[dict[str, Any]] = []
        if missing_required_rows:
            dropped_reason_counts["missing_required_values"] = missing_required_rows
            warnings.append(
                f"Se descartaron {missing_required_rows} filas de mercado con columnas obligatorias vacias o no parseables."
            )
        if normalized.empty:
            raise ValueError("El dataframe de mercado queda vacio tras normalizar columnas y tipos")

        duplicate_mask = normalized.duplicated(subset=["Ticker", "Date"], keep="last")
        duplicate_rows_resolved = int(duplicate_mask.sum())
        if duplicate_rows_resolved:
            duplicate_examples = (
                normalized.loc[duplicate_mask, ["Ticker", "Date"]]
                .astype(str)
                .drop_duplicates()
                .head(self.MAX_EXAMPLES)
                .to_dict(orient="records")
            )
            warnings.append(
                f"Se resolvieron {duplicate_rows_resolved} filas duplicadas por ticker/fecha conservando la ultima observacion."
            )
            if len(dropped_examples) < self.MAX_EXAMPLES:
                dropped_examples.extend(
                    {
                        "ticker": example["Ticker"],
                        "date": example["Date"],
                        "reason": "duplicate_ticker_date_resolved_keep_last",
                    }
                    for example in duplicate_examples
                )
            dropped_reason_counts["duplicate_ticker_date_resolved_keep_last"] = duplicate_rows_resolved
            normalized = normalized.loc[~duplicate_mask].copy()

        normalized = normalized.sort_values(["Ticker", "Date"]).reset_index(drop=True)
        cleaned_rows: list[dict[str, Any]] = []
        dropped_rows = 0
        repaired_rows = 0
        for row in normalized.to_dict(orient="records"):
            cleaned_row, row_reasons, drop_reason = self._clean_market_row(row)
            if drop_reason is not None:
                dropped_rows += 1
                dropped_reason_counts[drop_reason] = dropped_reason_counts.get(drop_reason, 0) + 1
                if len(dropped_examples) < self.MAX_EXAMPLES:
                    dropped_examples.append(
                        {
                            "ticker": str(row["Ticker"]),
                            "date": pd.Timestamp(row["Date"]).isoformat(),
                            "reason": drop_reason,
                        }
                    )
                continue
            cleaned_rows.append(cleaned_row)
            if row_reasons:
                repaired_rows += 1
                for reason in row_reasons:
                    repaired_reason_counts[reason] = repaired_reason_counts.get(reason, 0) + 1
                if len(repaired_examples) < self.MAX_EXAMPLES:
                    repaired_examples.append(
                        {
                            "ticker": str(cleaned_row["Ticker"]),
                            "date": pd.Timestamp(cleaned_row["Date"]).isoformat(),
                            "reasons": list(row_reasons),
                        }
                    )

        if not cleaned_rows:
            raise ValueError("El dataframe de mercado queda vacio tras aplicar los controles de calidad OHLC")

        cleaned = pd.DataFrame(cleaned_rows, columns=list(self.REQUIRED_COLUMNS))
        if repaired_rows:
            warnings.append(
                f"Se repararon {repaired_rows} filas de mercado ajustando la envolvente OHLC minima necesaria."
            )
        if dropped_rows:
            warnings.append(
                f"Se descartaron {dropped_rows} filas de mercado invalidas tras los controles de calidad OHLC."
            )
        quality_report = MarketDataQualityReport(
            input_rows=input_rows,
            output_rows=int(len(cleaned)),
            repaired_rows=repaired_rows,
            dropped_rows=int(missing_required_rows + duplicate_rows_resolved + dropped_rows),
            duplicate_rows_resolved=duplicate_rows_resolved,
            repaired_reason_counts=dict(repaired_reason_counts),
            dropped_reason_counts=dict(dropped_reason_counts),
            repaired_examples=tuple(repaired_examples),
            dropped_examples=tuple(dropped_examples),
        )
        return cleaned, quality_report, tuple(warnings)

    def _clean_market_row(
        self,
        row: dict[str, Any],
    ) -> tuple[dict[str, Any], tuple[str, ...], str | None]:
        open_price = float(row["Open"])
        high_price = float(row["High"])
        low_price = float(row["Low"])
        close_price = float(row["Close"])
        volume = float(row["Volume"])

        numeric_values = {
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume,
        }
        if any(not math.isfinite(value) for value in numeric_values.values()):
            return row, (), "non_finite_numeric"
        if min(open_price, high_price, low_price, close_price) <= 0.0:
            return row, (), "non_positive_ohlc"
        if volume < 0.0:
            return row, (), "negative_volume"

        envelope_high = max(open_price, high_price, low_price, close_price)
        envelope_low = min(open_price, high_price, low_price, close_price)
        repaired_reasons: list[str] = []

        if not math.isclose(high_price, envelope_high, rel_tol=self._ohlc_repair_rel_tol, abs_tol=self._ohlc_repair_abs_tol):
            repaired_reasons.append("high_envelope_adjusted")
        elif high_price != envelope_high:
            repaired_reasons.append("high_rounding_adjusted")
        if not math.isclose(low_price, envelope_low, rel_tol=self._ohlc_repair_rel_tol, abs_tol=self._ohlc_repair_abs_tol):
            repaired_reasons.append("low_envelope_adjusted")
        elif low_price != envelope_low:
            repaired_reasons.append("low_rounding_adjusted")

        cleaned = dict(row)
        cleaned["High"] = float(envelope_high)
        cleaned["Low"] = float(envelope_low)
        return cleaned, tuple(repaired_reasons), None

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
