from __future__ import annotations

from datetime import datetime, timedelta

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
