from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Iterable, List, Optional, Tuple, Union
from uuid import uuid4

import pandas as pd
import requests
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

try:
    from curl_cffi import requests as curl_requests
except Exception:
    curl_requests = None


PROVIDER_NAME = "TFT.YFinanceProvider"
PROVIDER_VERSION = "1.1.0"

EXPORT_COLUMNS = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Volume",
    "Dividends",
    "Stock Splits",
]

ALLOWED_INTERVALS = {
    "1m",
    "2m",
    "5m",
    "15m",
    "30m",
    "60m",
    "90m",
    "1h",
    "1d",
    "5d",
    "1wk",
    "1mo",
    "3mo",
}

INTERVAL_ALIASES = {
    "d": "1d",
    "1day": "1d",
    "day": "1d",
    "daily": "1d",
    "w": "1wk",
    "1w": "1wk",
    "week": "1wk",
    "weekly": "1wk",
    "m": "1mo",
    "1mth": "1mo",
    "month": "1mo",
    "monthly": "1mo",
    "h": "1h",
    "60m": "1h",
}

INTRADAY_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "90m", "1h"}
DAILY_LIKE_INTERVALS = {"1d", "5d", "1wk", "1mo", "3mo"}

logger = logging.getLogger(__name__)
PROXY_ENV_KEYS = (
    "ALL_PROXY",
    "all_proxy",
    "HTTP_PROXY",
    "http_proxy",
    "HTTPS_PROXY",
    "https_proxy",
    "GIT_HTTP_PROXY",
    "GIT_HTTPS_PROXY",
)
PROXY_ENV_LOCK = RLock()


class YFinanceProviderError(Exception):
    """Error base del provider."""


class ProviderConfigurationError(YFinanceProviderError):
    """Configuracion invalida del provider."""


class RequestValidationError(YFinanceProviderError):
    """Peticion semantica invalida."""


class EmptyDatasetError(YFinanceProviderError):
    """Yahoo no devolvio filas utiles."""


class DownloadFailureError(YFinanceProviderError):
    """Todos los backends y reintentos han fallado."""


@dataclass
class DownloadAttempt:
    attempt_number: int
    backend: str
    interval: str
    start: Optional[str]
    end: Optional[str]
    duration_seconds: float
    success: bool
    rows: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FetchMetadata:
    provider_name: str
    provider_version: str
    source: str
    request_id: str
    requested_symbol: str
    resolved_symbol: str
    requested_interval: str
    resolved_interval: str
    requested_start: Optional[str]
    requested_end: Optional[str]
    effective_start: Optional[str]
    effective_end: Optional[str]
    actual_start: Optional[str]
    actual_end: Optional[str]
    extracted_at_utc: str
    auto_adjust: bool
    actions: bool
    repair: bool
    backend_used: Optional[str] = None
    chunked: bool = False
    chunk_count: int = 0
    row_count: int = 0
    warnings: List[str] = field(default_factory=list)
    attempts: List[DownloadAttempt] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["attempts"] = [attempt.to_dict() for attempt in self.attempts]
        return payload


@dataclass
class FetchResult:
    symbol: str
    data: pd.DataFrame
    metadata: FetchMetadata


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _ts_to_iso(ts: Optional[pd.Timestamp]) -> Optional[str]:
    if ts is None:
        return None
    return ts.isoformat()


def _normalize_symbol(symbol: str) -> str:
    value = str(symbol or "").strip().upper()
    if not value:
        raise RequestValidationError("El ticker no puede estar vacio.")
    return value


def _normalize_interval(interval: Optional[str]) -> Tuple[str, str]:
    raw = "1d" if interval is None else str(interval).strip()
    if not raw:
        raw = "1d"

    lowered = raw.lower().replace(" ", "")
    normalized = INTERVAL_ALIASES.get(lowered, lowered)
    if normalized not in ALLOWED_INTERVALS:
        valid = ", ".join(sorted(ALLOWED_INTERVALS))
        raise RequestValidationError(f"Intervalo no soportado {raw!r}. Valores permitidos: {valid}.")
    return raw, normalized


def _to_naive_utc(ts: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if ts is None or ts == "":
        return None
    out = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(out):
        raise RequestValidationError(f"Marca temporal invalida: {ts!r}")
    return out.tz_convert(None)


def _validate_date_range(start: Optional[pd.Timestamp], end: Optional[pd.Timestamp]) -> None:
    if start is not None and end is not None and start >= end:
        raise RequestValidationError(
            f"Rango temporal invalido: start ({start.isoformat()}) debe ser anterior a end ({end.isoformat()})."
        )


def _is_intraday(interval: str) -> bool:
    return interval in INTRADAY_INTERVALS


def _is_daily_like(interval: str) -> bool:
    return interval in DAILY_LIKE_INTERVALS


def _intraday_chunk_timedelta(max_lookback_days: int) -> pd.Timedelta:
    safe_days = max(1, max_lookback_days - 1)
    return pd.Timedelta(days=safe_days)


def _build_intraday_chunks(start: pd.Timestamp, end: pd.Timestamp, max_lookback_days: int) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if start >= end:
        return []

    delta = _intraday_chunk_timedelta(max_lookback_days)
    chunks: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + delta, end)
        chunks.append((cursor, chunk_end))
        cursor = chunk_end
    return chunks


def _ensure_datetime_index(df: pd.DataFrame, preserve_local_dates: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]

    if getattr(out.index, "tz", None) is not None:
        if preserve_local_dates:
            out.index = out.index.tz_localize(None)
        else:
            out.index = out.index.tz_convert(None)

    out = out.sort_index()
    if getattr(out.index, "has_duplicates", False):
        out = out[~out.index.duplicated(keep="last")]
    return out


def _flatten_columns_if_needed(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    if hasattr(out, "columns") and getattr(out.columns, "nlevels", 1) > 1:
        try:
            level0 = out.columns.get_level_values(0)
            if symbol in level0:
                out = out[symbol]
            else:
                out.columns = out.columns.get_level_values(-1)
        except Exception:
            out.columns = out.columns.get_level_values(-1)
    return out


def _clean_df(df: pd.DataFrame, preserve_local_dates: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = _ensure_datetime_index(df, preserve_local_dates=preserve_local_dates)
    if out.empty:
        return out

    if "Close" not in out.columns and "Adj Close" in out.columns:
        out["Close"] = out["Adj Close"]

    required_ohlc = [column for column in ["Open", "High", "Low", "Close"] if column in out.columns]
    if required_ohlc:
        out = out.dropna(subset=required_ohlc)
    return out


def _calendarize_daily(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if not _is_daily_like(interval):
        return _ensure_datetime_index(df, preserve_local_dates=False)

    out = _ensure_datetime_index(df, preserve_local_dates=True)
    if out.empty:
        return out

    out.index = out.index.normalize()
    if getattr(out.index, "has_duplicates", False):
        aggregation: Dict[str, str] = {}
        for column in out.columns:
            lowered = column.lower()
            if lowered == "open":
                aggregation[column] = "first"
            elif lowered == "high":
                aggregation[column] = "max"
            elif lowered == "low":
                aggregation[column] = "min"
            elif lowered in {"close", "adj close"}:
                aggregation[column] = "last"
            elif lowered == "volume":
                aggregation[column] = "sum"
            elif lowered in {"dividends", "stock splits"}:
                aggregation[column] = "sum"
            else:
                aggregation[column] = "last"
        out = out.groupby(out.index).agg(aggregation).sort_index()
        out = out[~out.index.duplicated(keep="last")]
    return out


def _select_export_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=EXPORT_COLUMNS)

    out = df.copy()
    if "Adj Close" not in out.columns and "Close" in out.columns:
        out["Adj Close"] = out["Close"]
    if "Dividends" not in out.columns:
        out["Dividends"] = 0.0
    if "Stock Splits" not in out.columns:
        out["Stock Splits"] = 0.0
    if "Volume" not in out.columns:
        out["Volume"] = 0

    out = out.reset_index()
    date_column = "Date" if "Date" in out.columns else out.columns[0]
    rename_map = {
        date_column: "Date",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Adj Close": "Adj Close",
        "Volume": "Volume",
        "Dividends": "Dividends",
        "Stock Splits": "Stock Splits",
    }
    out = out.rename(columns=rename_map)

    for column in EXPORT_COLUMNS:
        if column not in out.columns:
            if column in {"Dividends", "Stock Splits"}:
                out[column] = 0.0
            elif column == "Volume":
                out[column] = 0
            else:
                out[column] = pd.NA

    out = out[EXPORT_COLUMNS].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out = out.dropna(subset=["Date"])

    numeric_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Adj Close",
        "Volume",
        "Dividends",
        "Stock Splits",
    ]
    for column in numeric_columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")

    out = out.dropna(subset=["Open", "High", "Low", "Close"])
    out["Volume"] = out["Volume"].fillna(0)
    out["Dividends"] = out["Dividends"].fillna(0.0)
    out["Stock Splits"] = out["Stock Splits"].fillna(0.0)
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def _empty_export_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXPORT_COLUMNS)


def _attach_metadata(df: pd.DataFrame, metadata: FetchMetadata) -> pd.DataFrame:
    out = df.copy()
    out.attrs["provider_metadata"] = metadata.to_dict()
    out.attrs["provider_name"] = metadata.provider_name
    out.attrs["provider_version"] = metadata.provider_version
    return out


def _extract_actual_bounds(df: pd.DataFrame) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    if df is None or df.empty or "Date" not in df.columns:
        return None, None
    values = pd.to_datetime(df["Date"], errors="coerce").dropna()
    if values.empty:
        return None, None
    return values.min(), values.max()


class YFinanceProvider:
    """Provider de Yahoo Finance robusto y compatible con el esquema del proyecto."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        retries: int = 4,
        timeout: float = 10.0,
        min_delay: float = 0.25,
        max_intraday_lookback_days: int = 60,
        allow_partial_intraday: bool = False,
        cache_dir: Path | str | None = None,
        repair: bool = True,
        auto_reset_cookie_cache: bool = True,
        trust_env_proxies: bool = False,
    ) -> None:
        self.max_workers = int(max_workers)
        self.retries = int(retries)
        self.timeout = float(timeout)
        self.min_delay = float(min_delay)
        self.max_intraday_lookback_days = int(max_intraday_lookback_days)
        self.allow_partial_intraday = bool(allow_partial_intraday)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.repair = bool(repair)
        self.auto_reset_cookie_cache = bool(auto_reset_cookie_cache)
        self.trust_env_proxies = bool(trust_env_proxies)
        self.session_backend = "curl_cffi" if curl_requests is not None else "requests"

        if self.max_workers < 1:
            raise ProviderConfigurationError("max_workers debe ser >= 1.")
        if self.retries < 1:
            raise ProviderConfigurationError("retries debe ser >= 1.")
        if self.timeout <= 0:
            raise ProviderConfigurationError("timeout debe ser > 0.")
        if self.min_delay < 0:
            raise ProviderConfigurationError("min_delay debe ser >= 0.")
        if self.max_intraday_lookback_days < 1:
            raise ProviderConfigurationError("max_intraday_lookback_days debe ser >= 1.")

        self._configure_cache()

    def _configure_cache(self) -> None:
        if self.cache_dir is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            yf.set_tz_cache_location(str(self.cache_dir))
        except Exception as exc:
            logger.warning("No se ha podido fijar la cache de yfinance en %s: %s", self.cache_dir, exc)

    def _build_session(self) -> requests.Session:
        if curl_requests is not None:
            session = curl_requests.Session(impersonate="chrome")
        else:
            session = requests.Session()
        if hasattr(session, "trust_env"):
            session.trust_env = self.trust_env_proxies
        if not self.trust_env_proxies and hasattr(session, "proxies"):
            session.proxies = {}
        session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36",
                "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
            }
        )
        return session

    @contextmanager
    def _proxy_env_scope(self):
        if self.trust_env_proxies:
            yield
            return
        removed_values: Dict[str, str] = {}
        with PROXY_ENV_LOCK:
            for key in PROXY_ENV_KEYS:
                if key in os.environ:
                    removed_values[key] = os.environ.pop(key)
            try:
                yield
            finally:
                os.environ.update(removed_values)

    def _resolve_cookie_cache_dir(self) -> Optional[Path]:
        if self.cache_dir is not None:
            return self.cache_dir
        try:
            from yfinance.cache import _CookieDBManager

            location = _CookieDBManager.get_location()
            return Path(location) if location else None
        except Exception:
            return None

    def _reset_cookie_runtime(self) -> None:
        try:
            from yfinance.cache import _CookieCacheManager, _CookieDBManager, get_cookie_cache

            try:
                cookie_cache = get_cookie_cache()
                cookie_cache.store("basic", None)
                cookie_cache.store("csrf", None)
            finally:
                _CookieDBManager.close_db()
                _CookieDBManager._db = None
                _CookieCacheManager._Cookie_cache = None
        except Exception as exc:
            logger.warning("No se ha podido reiniciar la base de cookies de yfinance por API: %s", exc)

        cache_dir = self._resolve_cookie_cache_dir()
        if cache_dir is not None:
            for name in ("cookies.db", "cookies.db-shm", "cookies.db-wal"):
                target = cache_dir / name
                try:
                    if target.exists():
                        target.unlink()
                except Exception as exc:
                    logger.warning("No se ha podido limpiar %s: %s", target, exc)

        try:
            from yfinance.data import SingletonMeta, YfData

            SingletonMeta._instances.pop(YfData, None)
        except Exception as exc:
            logger.warning("No se ha podido reiniciar el estado interno de yfinance: %s", exc)

    @staticmethod
    def _exception_chain(exc: Exception) -> List[Exception]:
        chain: List[Exception] = []
        current: Optional[BaseException] = exc
        seen = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, Exception):
                chain.append(current)
            current = current.__cause__ or current.__context__
        return chain

    def _is_rate_limit_error(self, exc: Exception) -> bool:
        for item in self._exception_chain(exc):
            if isinstance(item, YFRateLimitError):
                return True
            message = str(item).lower()
            if "too many requests" in message or "rate limit" in message or "429" in message:
                return True
        return False

    def _download_via_ticker(
        self,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        with self._proxy_env_scope():
            ticker = yf.Ticker(symbol, session=self._build_session())
            kwargs = {
                "start": start,
                "end": end,
                "interval": interval,
                "auto_adjust": auto_adjust,
                "actions": actions,
                "repair": self.repair,
            }
            try:
                return ticker.history(timeout=self.timeout, **kwargs)
            except TypeError:
                try:
                    return ticker.history(**kwargs)
                except TypeError:
                    kwargs.pop("repair", None)
                    return ticker.history(**kwargs)

    def _download_via_download(
        self,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        with self._proxy_env_scope():
            kwargs = {
                "tickers": symbol,
                "start": start,
                "end": end,
                "interval": interval,
                "auto_adjust": auto_adjust,
                "actions": actions,
                "progress": False,
                "threads": False,
                "keepna": True,
                "repair": self.repair,
                "session": self._build_session(),
            }
            try:
                raw = yf.download(**kwargs)
            except TypeError:
                try:
                    kwargs.pop("repair", None)
                    raw = yf.download(**kwargs)
                except TypeError:
                    kwargs.pop("keepna", None)
                    raw = yf.download(**kwargs)
            return _flatten_columns_if_needed(raw, symbol)

    def _prepare_request_window(
        self,
        interval: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
    ) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], List[str]]:
        warnings: List[str] = []
        now = pd.Timestamp.now(tz="UTC").tz_convert(None)
        effective_end = end
        effective_start = start

        if effective_end is not None and effective_end > now:
            warnings.append(
                "La fecha final solicitada estaba en el futuro respecto a UTC y se ha truncado al instante actual."
            )
            effective_end = now

        _validate_date_range(effective_start, effective_end)

        if not _is_intraday(interval):
            return effective_start, effective_end, warnings

        effective_end = effective_end or now

        if effective_start is None:
            effective_start = effective_end - pd.Timedelta(days=self.max_intraday_lookback_days - 1)
            warnings.append(
                "La peticion intradia no incluia start y se ha inferido una ventana segura segun la configuracion del provider."
            )

        _validate_date_range(effective_start, effective_end)

        oldest_allowed = effective_end - pd.Timedelta(days=self.max_intraday_lookback_days)
        if effective_start < oldest_allowed:
            if not self.allow_partial_intraday:
                raise RequestValidationError(
                    "La peticion intradia excede la ventana maxima permitida por la configuracion del provider."
                )
            warnings.append(
                "La peticion intradia excedia la ventana maxima y ha sido truncada al rango permitido."
            )
            effective_start = oldest_allowed

        return effective_start, effective_end, warnings

    def _normalize_raw_history(self, raw: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        preserve_local_dates = _is_daily_like(interval)
        frame = _flatten_columns_if_needed(raw, symbol)
        frame = _clean_df(frame, preserve_local_dates=preserve_local_dates)
        frame = _calendarize_daily(frame, interval)
        frame = _select_export_columns(frame)

        if frame.empty:
            return frame

        if frame["Date"].duplicated().any():
            frame = frame.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        return frame.sort_values("Date").reset_index(drop=True)

    def _merge_chunk_frames(self, frames: List[pd.DataFrame], interval: str) -> pd.DataFrame:
        if not frames:
            return _empty_export_frame()

        merged = pd.concat(frames, axis=0, ignore_index=True)
        merged["Date"] = pd.to_datetime(merged["Date"], errors="coerce")
        merged = merged.dropna(subset=["Date"])

        if _is_daily_like(interval):
            merged["Date"] = merged["Date"].dt.normalize()
            merged = (
                merged.sort_values("Date")
                .groupby("Date", as_index=False)
                .agg(
                    {
                        "Open": "first",
                        "High": "max",
                        "Low": "min",
                        "Close": "last",
                        "Adj Close": "last",
                        "Volume": "sum",
                        "Dividends": "sum",
                        "Stock Splits": "sum",
                    }
                )
            )
        else:
            merged = merged.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

        return merged[EXPORT_COLUMNS].copy()

    def _download_window(
        self,
        backend: str,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> pd.DataFrame:
        if backend == "ticker":
            raw = self._download_via_ticker(symbol, start, end, interval, auto_adjust, actions)
        elif backend == "download":
            raw = self._download_via_download(symbol, start, end, interval, auto_adjust, actions)
        else:
            raise ProviderConfigurationError(f"Backend no soportado: {backend!r}")
        return self._normalize_raw_history(raw, symbol, interval)

    def _download_range(
        self,
        backend: str,
        symbol: str,
        start: Optional[pd.Timestamp],
        end: Optional[pd.Timestamp],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> Tuple[pd.DataFrame, bool, int]:
        if _is_intraday(interval) and start is not None and end is not None:
            chunks = _build_intraday_chunks(start, end, self.max_intraday_lookback_days)
        else:
            chunks = [(start, end)]

        frames: List[pd.DataFrame] = []
        for chunk_start, chunk_end in chunks:
            frame = self._download_window(
                backend=backend,
                symbol=symbol,
                start=chunk_start,
                end=chunk_end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            )
            if not frame.empty:
                frames.append(frame)

        merged = self._merge_chunk_frames(frames, interval)
        return merged, len(chunks) > 1, len(chunks)

    def _build_metadata(
        self,
        requested_symbol: str,
        resolved_symbol: str,
        requested_interval: str,
        resolved_interval: str,
        requested_start: Optional[pd.Timestamp],
        requested_end: Optional[pd.Timestamp],
        effective_start: Optional[pd.Timestamp],
        effective_end: Optional[pd.Timestamp],
        auto_adjust: bool,
        actions: bool,
    ) -> FetchMetadata:
        return FetchMetadata(
            provider_name=PROVIDER_NAME,
            provider_version=PROVIDER_VERSION,
            source="yahoo_finance_via_yfinance",
            request_id=str(uuid4()),
            requested_symbol=requested_symbol,
            resolved_symbol=resolved_symbol,
            requested_interval=requested_interval,
            resolved_interval=resolved_interval,
            requested_start=_ts_to_iso(requested_start),
            requested_end=_ts_to_iso(requested_end),
            effective_start=_ts_to_iso(effective_start),
            effective_end=_ts_to_iso(effective_end),
            actual_start=None,
            actual_end=None,
            extracted_at_utc=_utcnow_iso(),
            auto_adjust=auto_adjust,
            actions=actions,
            repair=self.repair,
        )

    def _fetch_one_result(
        self,
        symbol: str,
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
        interval: str,
        auto_adjust: bool,
        actions: bool,
    ) -> FetchResult:
        requested_symbol = _normalize_symbol(symbol)
        requested_start = _to_naive_utc(start)
        requested_end = _to_naive_utc(end)
        requested_interval, resolved_interval = _normalize_interval(interval)
        effective_start, effective_end, warnings = self._prepare_request_window(
            resolved_interval,
            requested_start,
            requested_end,
        )
        metadata = self._build_metadata(
            requested_symbol=requested_symbol,
            resolved_symbol=requested_symbol,
            requested_interval=requested_interval,
            resolved_interval=resolved_interval,
            requested_start=requested_start,
            requested_end=requested_end,
            effective_start=effective_start,
            effective_end=effective_end,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        metadata.warnings.extend(warnings)

        if self.min_delay > 0:
            time.sleep(self.min_delay * random.uniform(0.5, 1.5))

        backends = ("ticker", "download")
        last_error: Optional[Exception] = None
        reset_done = False

        for attempt_number in range(1, self.retries + 1):
            for backend in backends:
                started = time.perf_counter()
                try:
                    frame, chunked, chunk_count = self._download_range(
                        backend=backend,
                        symbol=requested_symbol,
                        start=effective_start,
                        end=effective_end,
                        interval=resolved_interval,
                        auto_adjust=auto_adjust,
                        actions=actions,
                    )
                    duration = time.perf_counter() - started
                    metadata.attempts.append(
                        DownloadAttempt(
                            attempt_number=attempt_number,
                            backend=backend,
                            interval=resolved_interval,
                            start=_ts_to_iso(effective_start),
                            end=_ts_to_iso(effective_end),
                            duration_seconds=round(duration, 6),
                            success=not frame.empty,
                            rows=int(len(frame)),
                            error=None if not frame.empty else "dataframe vacio",
                        )
                    )

                    if frame.empty:
                        last_error = EmptyDatasetError(
                            f"Yahoo no devolvio filas utiles para {requested_symbol} en el intervalo {resolved_interval}."
                        )
                        continue

                    actual_start, actual_end = _extract_actual_bounds(frame)
                    metadata.actual_start = _ts_to_iso(actual_start)
                    metadata.actual_end = _ts_to_iso(actual_end)
                    metadata.backend_used = backend
                    metadata.chunked = chunked
                    metadata.chunk_count = chunk_count
                    metadata.row_count = int(len(frame))
                    frame = _attach_metadata(frame, metadata)
                    return FetchResult(symbol=requested_symbol, data=frame, metadata=metadata)

                except Exception as exc:
                    duration = time.perf_counter() - started
                    last_error = exc
                    metadata.attempts.append(
                        DownloadAttempt(
                            attempt_number=attempt_number,
                            backend=backend,
                            interval=resolved_interval,
                            start=_ts_to_iso(effective_start),
                            end=_ts_to_iso(effective_end),
                            duration_seconds=round(duration, 6),
                            success=False,
                            rows=0,
                            error=str(exc),
                        )
                    )
                    if self.auto_reset_cookie_cache and not reset_done and self._is_rate_limit_error(exc):
                        reset_done = True
                        metadata.warnings.append(
                            "Se ha detectado un posible bloqueo por cookie/crumb de yfinance y se ha reiniciado la cache persistente."
                        )
                        self._reset_cookie_runtime()
                        time.sleep(max(1.0, self.min_delay))

            if attempt_number < self.retries:
                sleep_seconds = (2 ** (attempt_number - 1)) * 0.8 + random.random() * 0.4
                time.sleep(sleep_seconds)

        if last_error is None:
            last_error = DownloadFailureError(f"Fallo de descarga desconocido para {requested_symbol}.")

        raise DownloadFailureError(
            f"No se ha podido descargar {requested_symbol} tras {self.retries} reintentos."
        ) from last_error

    def _failure_result(
        self,
        symbol: str,
        start: Optional[Union[str, pd.Timestamp]],
        end: Optional[Union[str, pd.Timestamp]],
        interval: str,
        auto_adjust: bool,
        actions: bool,
        error: Exception,
    ) -> FetchResult:
        requested_symbol = str(symbol or "").strip().upper()
        requested_start = _to_naive_utc(start) if start not in (None, "") else None
        requested_end = _to_naive_utc(end) if end not in (None, "") else None
        try:
            requested_interval, resolved_interval = _normalize_interval(interval)
        except Exception:
            requested_interval = str(interval)
            resolved_interval = str(interval)

        metadata = self._build_metadata(
            requested_symbol=requested_symbol,
            resolved_symbol=requested_symbol,
            requested_interval=requested_interval,
            resolved_interval=resolved_interval,
            requested_start=requested_start,
            requested_end=requested_end,
            effective_start=requested_start,
            effective_end=requested_end,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        metadata.warnings.append(f"El ticker ha fallado en descarga por lotes: {error}")
        metadata.attempts.append(
            DownloadAttempt(
                attempt_number=0,
                backend="n/a",
                interval=resolved_interval,
                start=_ts_to_iso(requested_start),
                end=_ts_to_iso(requested_end),
                duration_seconds=0.0,
                success=False,
                rows=0,
                error=str(error),
            )
        )
        frame = _attach_metadata(_empty_export_frame(), metadata)
        return FetchResult(symbol=requested_symbol, data=frame, metadata=metadata)

    def get_history_bundle(
        self,
        symbols: Union[str, Iterable[str]],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        actions: bool = False,
    ) -> Union[FetchResult, Dict[str, FetchResult]]:
        if isinstance(symbols, str):
            return self._fetch_one_result(
                symbol=symbols,
                start=start,
                end=end,
                interval=interval,
                auto_adjust=auto_adjust,
                actions=actions,
            )

        normalized_symbols: List[str] = []
        seen = set()
        for item in symbols:
            symbol = _normalize_symbol(str(item))
            if symbol not in seen:
                normalized_symbols.append(symbol)
                seen.add(symbol)

        if not normalized_symbols:
            return {}

        results: Dict[str, FetchResult] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._fetch_one_result,
                    symbol,
                    start,
                    end,
                    interval,
                    auto_adjust,
                    actions,
                ): symbol
                for symbol in normalized_symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as exc:
                    logger.warning("Fallo en la descarga por lotes de %s: %s", symbol, exc)
                    results[symbol] = self._failure_result(
                        symbol=symbol,
                        start=start,
                        end=end,
                        interval=interval,
                        auto_adjust=auto_adjust,
                        actions=actions,
                        error=exc,
                    )
        return results

    def get_history(
        self,
        symbols: Union[str, Iterable[str]],
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        interval: str = "1d",
        auto_adjust: bool = True,
        actions: bool = False,
    ) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        bundle = self.get_history_bundle(
            symbols=symbols,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=actions,
        )
        if isinstance(bundle, FetchResult):
            return bundle.data
        return {symbol: result.data for symbol, result in bundle.items()}

    def get_provider_info(self) -> dict:
        return {
            "provider_name": PROVIDER_NAME,
            "provider_version": PROVIDER_VERSION,
            "yfinance_version": getattr(yf, "__version__", "unknown"),
            "max_workers": self.max_workers,
            "retries": self.retries,
            "timeout": self.timeout,
            "min_delay": self.min_delay,
            "max_intraday_lookback_days": self.max_intraday_lookback_days,
            "allow_partial_intraday": self.allow_partial_intraday,
            "repair": self.repair,
            "session_backend": self.session_backend,
            "trust_env_proxies": self.trust_env_proxies,
            "export_columns": list(EXPORT_COLUMNS),
        }
