from __future__ import annotations

from datetime import datetime, timedelta
import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from .runtime_config import ConfigManager
from .utils.universe_integrity import UniverseIntegrityReport, build_universe_integrity_report
from .utils.yfinance_provider import FetchResult, YFinanceProvider

logger = logging.getLogger(__name__)

FetchUniverseReport = UniverseIntegrityReport


class FreshDataRequiredError(RuntimeError):
    pass


class DataFetcher:
    def __init__(self, config_manager: ConfigManager, years: int):
        self.config = config_manager.config
        self.years = years
        self.tickers_file = Path(self.config["data"]["tickers_file"])
        self.raw_data_path = Path(self.config["data"]["raw_data_path"])
        provider_config = self.config.get("data_fetch", {})
        cache_dir_value = provider_config.get("yfinance_cache_dir")
        self.yfinance_cache_dir = Path(cache_dir_value) if cache_dir_value else None
        self.extra_days = 50
        self.max_workers = min(8, max(1, int(provider_config.get("max_workers", 1))))
        self.enable_sector_lookup = bool(provider_config.get("use_yfinance_sector_lookup", False))
        self._sector_cache: dict[str, str] = {}
        self._configure_yfinance_cache()
        self.provider = YFinanceProvider(
            max_workers=self.max_workers,
            retries=int(provider_config.get("retries", 4)),
            timeout=float(provider_config.get("timeout", 10.0)),
            min_delay=float(provider_config.get("min_delay", 0.35)),
            max_intraday_lookback_days=int(provider_config.get("max_intraday_lookback_days", 60)),
            allow_partial_intraday=bool(provider_config.get("allow_partial_intraday", False)),
            cache_dir=self.yfinance_cache_dir,
            repair=bool(provider_config.get("repair", True)),
            auto_reset_cookie_cache=bool(provider_config.get("auto_reset_cookie_cache", True)),
            trust_env_proxies=bool(provider_config.get("trust_env_proxies", False)),
            session_backend=str(provider_config.get("session_backend", "requests")),
            rate_limit_cooldown_seconds=float(provider_config.get("rate_limit_cooldown_seconds", 8.0)),
            rate_limit_circuit_breaker_threshold=int(provider_config.get("rate_limit_circuit_breaker_threshold", 2)),
            rate_limit_circuit_breaker_seconds=float(provider_config.get("rate_limit_circuit_breaker_seconds", 30.0)),
        )

    def _configure_yfinance_cache(self) -> None:
        if self.yfinance_cache_dir is None:
            return
        self.yfinance_cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            yf.set_tz_cache_location(str(self.yfinance_cache_dir))
        except Exception as exc:
            logger.warning("No se ha podido fijar la cache de yfinance en %s: %s", self.yfinance_cache_dir, exc)

    def _normalize_sector(self, sector: str | None) -> str:
        if sector is None or pd.isna(sector):
            sector_value = "Unknown"
        else:
            sector_value = str(sector).strip() or "Unknown"
        if sector_value not in self.config["model"]["sectors"]:
            sector_value = "Unknown"
        return sector_value

    def _resolve_sector(self, ticker: str) -> str:
        if ticker in self._sector_cache:
            return self._sector_cache[ticker]

        if self.raw_data_path.exists():
            try:
                cached = pd.read_csv(self.raw_data_path, usecols=["Ticker", "Sector"])
                cached = cached[cached["Ticker"].astype(str).str.upper() == ticker]
                if not cached.empty:
                    sector_from_cache = self._normalize_sector(cached["Sector"].iloc[0])
                    self._sector_cache[ticker] = sector_from_cache
                    return sector_from_cache
            except Exception as exc:
                logger.debug("No se ha podido reutilizar el sector desde la cache local para %s: %s", ticker, exc)

        if not self.enable_sector_lookup:
            self._sector_cache[ticker] = "Unknown"
            return "Unknown"

        sector = "Unknown"
        try:
            info = yf.Ticker(ticker).info or {}
            sector = self._normalize_sector(info.get("sector"))
        except Exception as exc:
            logger.warning("No se ha podido resolver el sector de %s. Se usara Unknown. Detalle: %s", ticker, exc)

        self._sector_cache[ticker] = sector
        return sector

    @staticmethod
    def _empty_output_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"])

    @staticmethod
    def _describe_provider_failure(result: FetchResult | None) -> str:
        if not isinstance(result, FetchResult):
            return "Sin detalle del proveedor"

        metadata = result.metadata
        for attempt in reversed(metadata.attempts):
            if attempt.error:
                return attempt.error
        for warning in reversed(metadata.warnings):
            if warning:
                return warning
        return "Sin detalle del proveedor"

    @staticmethod
    def _collect_provider_errors(result: FetchResult | None) -> list[str]:
        if not isinstance(result, FetchResult):
            return []
        errors: list[str] = []
        for attempt in result.metadata.attempts:
            if attempt.error:
                errors.append(str(attempt.error))
        for warning in result.metadata.warnings:
            if warning:
                errors.append(str(warning))
        return list(dict.fromkeys(errors))

    @staticmethod
    def _attach_fetch_provenance(
        frame: pd.DataFrame,
        *,
        source: str,
        detail: str | None = None,
        errors: list[str] | None = None,
        backend_used: str | None = None,
    ) -> pd.DataFrame:
        annotated = frame.copy()
        annotated.attrs["fetch_provenance"] = {
            "source": source,
            "used_local_fallback": source == "local_cache",
            "detail": detail,
            "errors": list(dict.fromkeys(str(error) for error in (errors or []) if str(error))),
            "backend_used": backend_used,
        }
        return annotated

    def _resolve_single_fetch_fallback(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        detail: str,
        errors: list[str],
        allow_local_fallback: bool,
    ) -> pd.DataFrame:
        fallback = self._fallback_from_local_raw_data(ticker, start_date, end_date)
        if fallback.empty:
            raise FreshDataRequiredError(
                f"No se han podido obtener datos frescos para {ticker}. Detalle del proveedor: {detail}"
            )
        if not allow_local_fallback:
            raise FreshDataRequiredError(
                f"Se ha rechazado un fallback silencioso a cache local para {ticker}. "
                f"Detalle del proveedor: {detail}"
            )
        logger.warning("Se acepta un fallback explicito a cache local para %s. Detalle del proveedor: %s", ticker, detail)
        return self._attach_fetch_provenance(
            fallback,
            source="local_cache",
            detail=detail,
            errors=errors,
            backend_used=None,
        )

    def _fallback_from_local_raw_data(self, ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        if not self.raw_data_path.exists():
            return self._empty_output_frame()

        try:
            cached = pd.read_csv(self.raw_data_path, parse_dates=["Date"])
        except Exception as exc:
            logger.warning("No se ha podido leer la cache local %s: %s", self.raw_data_path, exc)
            return self._empty_output_frame()

        required = {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"}
        if not required.issubset(cached.columns):
            logger.warning("La cache local %s no tiene el esquema esperado y no se puede usar como fallback", self.raw_data_path)
            return self._empty_output_frame()

        cached["Date"] = pd.to_datetime(cached["Date"], utc=True, errors="coerce")
        cached = cached.dropna(subset=["Date"])
        cached["Date"] = cached["Date"].dt.tz_localize(None)
        start_ts = pd.Timestamp(self._normalize_date(start_date))
        end_ts = pd.Timestamp(self._normalize_date(end_date))
        filtered = cached[(cached["Ticker"] == ticker) & (cached["Date"] >= start_ts) & (cached["Date"] < end_ts)].copy()
        if filtered.empty:
            return self._empty_output_frame()

        logger.warning("Se usaran datos locales en cache para %s desde %s", ticker, self.raw_data_path)
        return filtered[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"]].reset_index(drop=True)

    def _prepare_output_frame(self, ticker: str, frame: pd.DataFrame) -> pd.DataFrame:
        if frame is None or frame.empty:
            return self._empty_output_frame()

        out = frame.copy()
        if "Date" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index()
            first_column = out.columns[0]
            if first_column != "Date":
                out = out.rename(columns={first_column: "Date"})

        if {"Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"}.issubset(out.columns):
            out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
            out = out.dropna(subset=["Date"])
            out["Date"] = out["Date"].dt.tz_localize(None)
            out["Ticker"] = out["Ticker"].astype(str).str.upper()
            out["Sector"] = out["Sector"].map(self._normalize_sector)
            return out[["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"]].reset_index(drop=True)

        out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
        out = out.dropna(subset=["Date"])
        out["Date"] = out["Date"].dt.tz_localize(None)
        out["Ticker"] = ticker
        out["Sector"] = self._resolve_sector(ticker)
        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume", "Ticker", "Sector"]
        return out[required_cols].reset_index(drop=True)

    def _load_tickers(self, region: str | None = None) -> list[str]:
        with open(self.tickers_file, "r", encoding="utf-8") as handle:
            tickers_config = yaml.safe_load(handle) or {}
        regions = tickers_config.get("tickers", {})
        if region:
            return list(regions.get(region, {}).keys())
        all_tickers: list[str] = []
        for region_tickers in regions.values():
            all_tickers.extend(region_tickers.keys())
        return list(dict.fromkeys(all_tickers))

    @staticmethod
    def _normalize_date(value: datetime) -> datetime:
        if hasattr(value, "tzinfo") and value.tzinfo is not None:
            return value.replace(tzinfo=None)
        return value

    def fetch_stock_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        allow_local_fallback: bool = False,
    ) -> pd.DataFrame:
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        adjusted_start_date = start_date - timedelta(days=self.extra_days)
        ticker = str(ticker).strip().upper()

        try:
            result = self.provider.get_history_bundle(
                symbols=ticker,
                start=adjusted_start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
            if not isinstance(result, FetchResult):
                logger.error("El provider devolvio un tipo inesperado para %s", ticker)
                return self._resolve_single_fetch_fallback(
                    ticker,
                    start_date,
                    end_date,
                    "El provider devolvio un tipo inesperado",
                    [],
                    allow_local_fallback,
                )

            df = result.data
            provider_errors = self._collect_provider_errors(result)
            backend_used = getattr(result.metadata, "backend_used", None)
            if df.empty:
                detail = self._describe_provider_failure(result)
                logger.warning("No hay datos descargados para %s. Detalle del proveedor: %s", ticker, detail)
                return self._resolve_single_fetch_fallback(
                    ticker,
                    start_date,
                    end_date,
                    detail,
                    provider_errors,
                    allow_local_fallback,
                )

            prepared = self._prepare_output_frame(ticker, df)
            prepared = prepared[prepared["Date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            if prepared.empty:
                detail = "Sin datos utiles tras filtrar el rango solicitado"
                logger.warning("No hay datos utiles para %s tras filtrar el rango solicitado", ticker)
                return self._resolve_single_fetch_fallback(
                    ticker,
                    start_date,
                    end_date,
                    detail,
                    provider_errors,
                    allow_local_fallback,
                )
            return self._attach_fetch_provenance(
                prepared,
                source="fresh_network",
                detail=None,
                errors=provider_errors,
                backend_used=backend_used,
            )
        except FreshDataRequiredError:
            raise
        except Exception as exc:
            logger.error("Error al descargar %s: %s", ticker, exc)
            return self._resolve_single_fetch_fallback(
                ticker,
                start_date,
                end_date,
                str(exc),
                [str(exc)],
                allow_local_fallback,
            )

    def fetch_many_stocks(self, tickers: list[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        combined, _ = self.fetch_many_stocks_with_report(tickers, start_date, end_date)
        return combined

    def fetch_many_stocks_with_report(self, tickers: list[str], start_date: datetime, end_date: datetime) -> tuple[pd.DataFrame, FetchUniverseReport]:
        if not tickers:
            return pd.DataFrame(), FetchUniverseReport([], [], [], {})

        results = []
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        adjusted_start_date = start_date - timedelta(days=self.extra_days)
        normalized_tickers = [str(ticker).strip().upper() for ticker in tickers if str(ticker).strip()]
        ticker_payloads: dict[str, dict] = {}

        try:
            bundle = self.provider.get_history_bundle(
                symbols=normalized_tickers,
                start=adjusted_start_date,
                end=end_date,
                interval="1d",
                auto_adjust=True,
                actions=False,
            )
        except Exception as exc:
            logger.error("Fallo general en la descarga por lotes: %s", exc)
            bundle = {}
            general_failure_detail = str(exc)
        else:
            general_failure_detail = None

        for ticker in normalized_tickers:
            result = bundle.get(ticker) if isinstance(bundle, dict) else None
            frame = result.data if isinstance(result, FetchResult) else pd.DataFrame()
            payload = {
                "fetch_result": result,
                "source": "missing",
                "backend_used": getattr(getattr(result, "metadata", None), "backend_used", None),
                "errors": self._collect_provider_errors(result),
                "discard_reason": None,
            }
            if frame.empty:
                fallback = self._fallback_from_local_raw_data(ticker, start_date, end_date)
                if fallback.empty:
                    detail = self._describe_provider_failure(result)
                    if detail == "Sin detalle del proveedor" and general_failure_detail:
                        detail = general_failure_detail
                    logger.warning("Se omite %s por falta de datos. Detalle del proveedor: %s", ticker, detail)
                    payload["discard_reason"] = detail
                    ticker_payloads[ticker] = payload
                    continue
                payload["frame"] = fallback
                payload["source"] = "local_cache"
                payload["discard_reason"] = self._describe_provider_failure(result)
                ticker_payloads[ticker] = payload
                continue

            prepared = self._prepare_output_frame(ticker, frame)
            prepared = prepared[prepared["Date"] >= pd.Timestamp(start_date)].reset_index(drop=True)
            if prepared.empty:
                fallback = self._fallback_from_local_raw_data(ticker, start_date, end_date)
                if fallback.empty:
                    detail = self._describe_provider_failure(result)
                    if detail == "Sin detalle del proveedor":
                        detail = "Sin datos utiles tras filtrar el rango solicitado"
                    logger.warning("Se omite %s por falta de datos utiles. Detalle: %s", ticker, detail)
                    payload["discard_reason"] = detail
                    ticker_payloads[ticker] = payload
                    continue
                payload["frame"] = fallback
                payload["source"] = "local_cache"
                payload["discard_reason"] = "sin_datos_utiles_tras_filtrado"
                ticker_payloads[ticker] = payload
                continue
            payload["frame"] = prepared
            payload["source"] = "fresh_network"
            ticker_payloads[ticker] = payload

        report = build_universe_integrity_report(self.config, normalized_tickers, ticker_payloads)
        for ticker in normalized_tickers:
            payload = ticker_payloads.get(ticker) or {}
            frame = payload.get("frame")
            if isinstance(frame, pd.DataFrame) and not frame.empty:
                results.append(frame)

        if not results:
            return pd.DataFrame(), report

        combined = pd.concat(results, ignore_index=True)
        combined["Sector"] = pd.Categorical(combined["Sector"], categories=self.config["model"]["sectors"], ordered=False)
        return combined, report

    def fetch_training_universe(self, tickers: list[str]) -> tuple[pd.DataFrame, FetchUniverseReport]:
        end_date = datetime.now().replace(tzinfo=None)
        start_date = end_date - timedelta(days=self.years * 365)
        combined, report = self.fetch_many_stocks_with_report(tickers, start_date, end_date)
        if combined.empty or not report.successful_tickers:
            return pd.DataFrame(), report
        filtered = combined[combined["Ticker"].astype(str).isin(report.successful_tickers)].reset_index(drop=True)
        return filtered, report

    def fetch_global_stocks(self, region: str | None = None) -> pd.DataFrame:
        end_date = datetime.now().replace(tzinfo=None)
        start_date = end_date - timedelta(days=self.years * 365)
        tickers = self.config.get("data", {}).get("tickers") or self._load_tickers(region)
        combined, _ = self.fetch_many_stocks_with_report(tickers, start_date, end_date)
        if combined.empty:
            logger.error("No se ha podido descargar ningun ticker")
            return combined
        return combined

    def fetch_stock_data_sync(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime,
        allow_local_fallback: bool = False,
    ) -> pd.DataFrame:
        return self.fetch_stock_data(ticker, start_date, end_date, allow_local_fallback=allow_local_fallback)


if __name__ == "__main__":
    config_manager = ConfigManager()
    fetcher = DataFetcher(config_manager, years=3)
    fetcher.fetch_global_stocks()
