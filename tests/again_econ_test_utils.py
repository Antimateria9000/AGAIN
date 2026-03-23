from __future__ import annotations

from datetime import datetime, timedelta, timezone

from again_econ.contracts import MarketBar, MarketFrame


def build_single_symbol_market(
    opens: list[float],
    closes: list[float] | None = None,
    *,
    symbol: str = "AAA",
    start: datetime | None = None,
) -> MarketFrame:
    if closes is None:
        closes = opens
    start_timestamp = start or datetime(2024, 1, 1)
    bars = []
    for index, (open_price, close_price) in enumerate(zip(opens, closes, strict=True)):
        timestamp = start_timestamp + timedelta(days=index)
        high_price = max(open_price, close_price) + 1.0
        low_price = min(open_price, close_price) - 1.0
        bars.append(
            MarketBar(
                instrument_id=symbol,
                timestamp=timestamp,
                open=float(open_price),
                high=float(high_price),
                low=float(low_price),
                close=float(close_price),
                volume=1_000.0,
            )
        )
    return MarketFrame(bars=tuple(bars))


def build_multi_symbol_market(
    symbol_to_opens: dict[str, list[float]],
    *,
    symbol_to_closes: dict[str, list[float]] | None = None,
    start: datetime | None = None,
) -> MarketFrame:
    all_bars = []
    for symbol, opens in sorted(symbol_to_opens.items()):
        closes = symbol_to_closes[symbol] if symbol_to_closes is not None else None
        market = build_single_symbol_market(opens, closes, symbol=symbol, start=start)
        all_bars.extend(market.bars)
    return MarketFrame(bars=tuple(sorted(all_bars, key=lambda bar: (bar.timestamp, bar.instrument_id))))


def aware_utc_datetime(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)
