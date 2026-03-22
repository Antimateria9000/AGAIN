from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


UNIVERSE_DECISIONS = {
    "ABORT",
    "DEGRADED_ALLOWED",
    "DEGRADED_FORBIDDEN",
    "CONTINUE_CLEAN",
}


@dataclass
class UniverseTickerIntegrity:
    ticker: str
    final_status: str
    source: str
    backend_used: str | None = None
    attempt_count: int = 0
    errors: list[str] = field(default_factory=list)
    rows_obtained: int = 0
    actual_start: str | None = None
    actual_end: str | None = None
    meets_minimum_rows: bool = False
    discard_reason: str | None = None
    is_anchor_ticker: bool = False
    trainable: bool = False
    used_fallback: bool = False
    freshness: str = "missing"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class UniverseIntegrityReport:
    requested_tickers: list[str]
    successful_tickers: list[str]
    discarded_tickers: list[str]
    discarded_details: dict[str, str]
    ticker_integrity: dict[str, UniverseTickerIntegrity] = field(default_factory=dict)
    fresh_tickers: list[str] = field(default_factory=list)
    fallback_tickers: list[str] = field(default_factory=list)
    coverage_abs: int = 0
    coverage_ratio: float = 0.0
    anchor_ticker: str | None = None
    anchor_present: bool = False
    anchor_usable: bool = False
    minimum_group_tickers_abs: int = 1
    minimum_group_coverage_ratio: float = 1.0
    minimum_rows_per_ticker: int = 1
    minimum_common_overlap_days: int = 1
    common_overlap_days: int = 0
    allow_degraded_universe: bool = False
    abort_if_anchor_missing: bool = True
    abort_if_too_many_fallback_tickers: bool = True
    abort_on_cached_backfill: bool = False
    maximum_fallback_tickers: int = 0
    maximum_fallback_ratio: float = 0.0
    decision: str = "ABORT"
    decision_reasons: list[str] = field(default_factory=list)
    training_allowed: bool = False
    can_promote_canonical: bool = False
    degraded: bool = False
    summary: str = ""

    def __post_init__(self) -> None:
        if self.coverage_abs <= 0:
            self.coverage_abs = len(self.successful_tickers)
        if self.coverage_ratio <= 0.0 and self.requested_tickers:
            self.coverage_ratio = float(self.coverage_abs / len(self.requested_tickers))
        if not self.summary:
            self.summary = (
                f"decision={self.decision} | requested={len(self.requested_tickers)} | usable={self.coverage_abs} | "
                f"fresh={len(self.fresh_tickers)} | fallback={len(self.fallback_tickers)} | "
                f"discarded={len(self.discarded_tickers)} | coverage={self.coverage_ratio:.2%}"
            )
        if (
            self.decision == "ABORT"
            and not self.decision_reasons
            and self.successful_tickers
            and not self.discarded_tickers
            and not self.ticker_integrity
        ):
            self.decision = "CONTINUE_CLEAN"
            self.training_allowed = True
            self.can_promote_canonical = True
            self.degraded = False

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ticker_integrity"] = {ticker: record.to_dict() for ticker, record in self.ticker_integrity.items()}
        return payload


def _safe_normalize_ticker(value: str | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _build_training_universe_policy(config: dict) -> dict[str, Any]:
    training_universe = dict(config.get("training_universe") or {})
    training_run = dict(config.get("training_run") or {})
    training_mode = str(training_run.get("mode") or training_universe.get("mode") or "").strip().lower()
    min_group_abs = int(training_universe.get("minimum_group_tickers_abs") or training_universe.get("minimum_group_tickers") or 1)
    if training_mode == "single_ticker":
        min_group_abs = 1
    minimum_rows_per_ticker = int(
        training_universe.get(
            "minimum_rows_per_ticker",
            int(config["model"]["max_encoder_length"]) + int(config["model"]["max_prediction_length"]) + 1,
        )
    )
    minimum_common_overlap_days = int(
        training_universe.get("minimum_common_overlap_days", int(config["model"]["max_prediction_length"]) + 1)
    )
    return {
        "minimum_group_tickers_abs": max(1, min_group_abs),
        "minimum_group_coverage_ratio": float(training_universe.get("minimum_group_coverage_ratio", 1.0)),
        "minimum_rows_per_ticker": max(1, minimum_rows_per_ticker),
        "minimum_common_overlap_days": max(1, minimum_common_overlap_days),
        "allow_degraded_universe": bool(training_universe.get("allow_degraded_universe", False)),
        "abort_if_anchor_missing": bool(training_universe.get("abort_if_anchor_missing", True)),
        "abort_if_too_many_fallback_tickers": bool(training_universe.get("abort_if_too_many_fallback_tickers", True)),
        "abort_on_cached_backfill": bool(training_universe.get("abort_on_cached_backfill", False)),
        "maximum_fallback_tickers": max(0, int(training_universe.get("maximum_fallback_tickers", 0))),
        "maximum_fallback_ratio": max(0.0, float(training_universe.get("maximum_fallback_ratio", 0.0))),
        "anchor_ticker": _safe_normalize_ticker(
            training_universe.get("anchor_ticker") or training_run.get("anchor_ticker")
        ),
    }


def _extract_overlap_days(ticker_frames: dict[str, pd.DataFrame], trainable_tickers: list[str]) -> int:
    normalized_sets: list[set[pd.Timestamp]] = []
    for ticker in trainable_tickers:
        frame = ticker_frames.get(ticker)
        if frame is None or frame.empty:
            continue
        dates = pd.to_datetime(frame["Date"], errors="coerce").dropna().dt.normalize().tolist()
        if not dates:
            continue
        normalized_sets.append(set(dates))
    if not normalized_sets:
        return 0
    overlap = set.intersection(*normalized_sets)
    return len(overlap)


def build_universe_integrity_report(
    config: dict,
    requested_tickers: list[str],
    ticker_payloads: dict[str, dict[str, Any]],
) -> UniverseIntegrityReport:
    requested = [_safe_normalize_ticker(ticker) for ticker in requested_tickers]
    requested = [ticker for ticker in requested if ticker]
    policy = _build_training_universe_policy(config)
    requested_count = len(requested)
    ticker_integrity: dict[str, UniverseTickerIntegrity] = {}
    successful_tickers: list[str] = []
    discarded_tickers: list[str] = []
    discarded_details: dict[str, str] = {}
    fresh_tickers: list[str] = []
    fallback_tickers: list[str] = []
    ticker_frames: dict[str, pd.DataFrame] = {}

    for ticker in requested:
        payload = dict(ticker_payloads.get(ticker) or {})
        frame = payload.get("frame")
        frame = frame.copy() if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        fetch_result = payload.get("fetch_result")
        metadata = getattr(fetch_result, "metadata", None)
        errors = [str(value) for value in payload.get("errors", []) if str(value)]
        if metadata is not None:
            errors.extend([str(attempt.error) for attempt in metadata.attempts if attempt.error])
        rows_obtained = int(len(frame))
        actual_start = None
        actual_end = None
        if not frame.empty and "Date" in frame.columns:
            dates = pd.to_datetime(frame["Date"], errors="coerce").dropna()
            if not dates.empty:
                actual_start = dates.min().isoformat()
                actual_end = dates.max().isoformat()
                ticker_frames[ticker] = frame
        elif metadata is not None:
            actual_start = metadata.actual_start
            actual_end = metadata.actual_end

        source = str(payload.get("source") or "missing")
        used_fallback = source == "local_cache"
        freshness = "cached" if used_fallback else ("fresh" if rows_obtained > 0 else "missing")
        backend_used = payload.get("backend_used") or getattr(metadata, "backend_used", None)
        meets_minimum_rows = rows_obtained >= policy["minimum_rows_per_ticker"]
        discard_reason = payload.get("discard_reason")
        final_status = "ok"
        trainable = rows_obtained > 0 and meets_minimum_rows

        if rows_obtained == 0:
            final_status = "discarded"
            discard_reason = discard_reason or "sin_datos"
            trainable = False
        elif not meets_minimum_rows:
            final_status = "discarded"
            discard_reason = discard_reason or f"filas_insuficientes<{policy['minimum_rows_per_ticker']}>"
            trainable = False
        elif used_fallback:
            final_status = "fallback_cache"

        record = UniverseTickerIntegrity(
            ticker=ticker,
            final_status=final_status,
            source=source,
            backend_used=backend_used,
            attempt_count=len(getattr(metadata, "attempts", []) or []),
            errors=list(dict.fromkeys(errors)),
            rows_obtained=rows_obtained,
            actual_start=actual_start,
            actual_end=actual_end,
            meets_minimum_rows=meets_minimum_rows,
            discard_reason=discard_reason,
            is_anchor_ticker=ticker == policy["anchor_ticker"] if policy["anchor_ticker"] else False,
            trainable=trainable,
            used_fallback=used_fallback,
            freshness=freshness,
        )
        ticker_integrity[ticker] = record

        if record.trainable:
            successful_tickers.append(ticker)
            if record.used_fallback:
                fallback_tickers.append(ticker)
            else:
                fresh_tickers.append(ticker)
        else:
            discarded_tickers.append(ticker)
            discarded_details[ticker] = discard_reason or (record.errors[-1] if record.errors else "sin_detalle")

    coverage_abs = len(successful_tickers)
    coverage_ratio = float(coverage_abs / requested_count) if requested_count else 0.0
    common_overlap_days = _extract_overlap_days(ticker_frames, successful_tickers)
    anchor_ticker = policy["anchor_ticker"]
    anchor_present = anchor_ticker in requested if anchor_ticker else False
    anchor_usable = anchor_ticker in successful_tickers if anchor_ticker else True

    decision_reasons: list[str] = []
    hard_failures: list[str] = []
    if coverage_abs == 0:
        hard_failures.append("No queda ningun ticker apto para entrenamiento.")
    if coverage_abs < policy["minimum_group_tickers_abs"]:
        hard_failures.append(
            f"El universo util ({coverage_abs}) queda por debajo del minimo absoluto ({policy['minimum_group_tickers_abs']})."
        )
    if coverage_ratio < policy["minimum_group_coverage_ratio"]:
        hard_failures.append(
            f"La cobertura del universo ({coverage_ratio:.2%}) queda por debajo del minimo configurado ({policy['minimum_group_coverage_ratio']:.2%})."
        )
    if anchor_ticker and policy["abort_if_anchor_missing"] and not anchor_usable:
        hard_failures.append(f"Falta el ticker ancla o no es apto para entrenar: {anchor_ticker}.")
    if coverage_abs > 1 and common_overlap_days < policy["minimum_common_overlap_days"]:
        hard_failures.append(
            f"El solape temporal comun ({common_overlap_days} dias) queda por debajo del minimo requerido ({policy['minimum_common_overlap_days']})."
        )
    fallback_ratio = float(len(fallback_tickers) / coverage_abs) if coverage_abs else 0.0
    if policy["abort_on_cached_backfill"] and fallback_tickers:
        hard_failures.append("La configuracion prohíbe entrenar con tickers recuperados desde cache local.")
    if policy["abort_if_too_many_fallback_tickers"]:
        if len(fallback_tickers) > policy["maximum_fallback_tickers"]:
            hard_failures.append(
                f"El numero de tickers por fallback local ({len(fallback_tickers)}) supera el maximo permitido ({policy['maximum_fallback_tickers']})."
            )
        if fallback_ratio > policy["maximum_fallback_ratio"]:
            hard_failures.append(
                f"La proporcion de tickers por fallback local ({fallback_ratio:.2%}) supera el maximo permitido ({policy['maximum_fallback_ratio']:.2%})."
            )

    degraded = bool(discarded_tickers or fallback_tickers)
    if hard_failures:
        decision = "ABORT"
        decision_reasons.extend(hard_failures)
        training_allowed = False
        can_promote_canonical = False
    elif degraded and not policy["allow_degraded_universe"]:
        decision = "DEGRADED_FORBIDDEN"
        decision_reasons.append("El universo esta degradado y la configuracion no permite continuar en modo degradado.")
        training_allowed = False
        can_promote_canonical = False
    elif degraded:
        decision = "DEGRADED_ALLOWED"
        decision_reasons.append("El universo esta degradado, pero cumple los minimos configurados para continuar de forma explicita.")
        training_allowed = True
        can_promote_canonical = False
    else:
        decision = "CONTINUE_CLEAN"
        training_allowed = True
        can_promote_canonical = True

    summary = (
        f"decision={decision} | requested={requested_count} | usable={coverage_abs} | "
        f"fresh={len(fresh_tickers)} | fallback={len(fallback_tickers)} | discarded={len(discarded_tickers)} | "
        f"coverage={coverage_ratio:.2%} | overlap={common_overlap_days}"
    )

    return UniverseIntegrityReport(
        requested_tickers=requested,
        successful_tickers=successful_tickers,
        discarded_tickers=discarded_tickers,
        discarded_details=discarded_details,
        ticker_integrity=ticker_integrity,
        fresh_tickers=fresh_tickers,
        fallback_tickers=fallback_tickers,
        coverage_abs=coverage_abs,
        coverage_ratio=coverage_ratio,
        anchor_ticker=anchor_ticker,
        anchor_present=anchor_present,
        anchor_usable=anchor_usable,
        minimum_group_tickers_abs=policy["minimum_group_tickers_abs"],
        minimum_group_coverage_ratio=policy["minimum_group_coverage_ratio"],
        minimum_rows_per_ticker=policy["minimum_rows_per_ticker"],
        minimum_common_overlap_days=policy["minimum_common_overlap_days"],
        common_overlap_days=common_overlap_days,
        allow_degraded_universe=policy["allow_degraded_universe"],
        abort_if_anchor_missing=policy["abort_if_anchor_missing"],
        abort_if_too_many_fallback_tickers=policy["abort_if_too_many_fallback_tickers"],
        abort_on_cached_backfill=policy["abort_on_cached_backfill"],
        maximum_fallback_tickers=policy["maximum_fallback_tickers"],
        maximum_fallback_ratio=policy["maximum_fallback_ratio"],
        decision=decision,
        decision_reasons=decision_reasons,
        training_allowed=training_allowed,
        can_promote_canonical=can_promote_canonical,
        degraded=degraded,
        summary=summary,
    )
