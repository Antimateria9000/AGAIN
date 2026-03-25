from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import torch

from again_econ.contracts import MarketFrame


def _timestamp_to_sortable_int(value: datetime) -> int:
    seconds_since_ordinal_start = (((value.toordinal() * 24) + value.hour) * 60 + value.minute) * 60 + value.second
    return seconds_since_ordinal_start * 1_000_000 + value.microsecond


@dataclass(frozen=True)
class TensorMarketFrame:
    device: torch.device
    timestamps: tuple[datetime, ...]
    instruments: tuple[str, ...]
    timestamp_values: torch.Tensor
    open_prices: torch.Tensor
    close_prices: torch.Tensor
    availability_mask: torch.Tensor
    instrument_index: dict[str, int]
    timestamp_index: dict[datetime, int]
    instrument_timestamp_values: dict[str, torch.Tensor]
    instrument_timestamp_indices: dict[str, torch.Tensor]

    @classmethod
    def from_market_frame(cls, market_frame: MarketFrame, *, device: torch.device) -> "TensorMarketFrame":
        timestamps = market_frame.timestamps()
        instruments = market_frame.instruments()
        timestamp_index = {timestamp: idx for idx, timestamp in enumerate(timestamps)}
        instrument_index = {instrument_id: idx for idx, instrument_id in enumerate(instruments)}
        open_prices = torch.full((len(timestamps), len(instruments)), float("nan"), dtype=torch.float64, device=device)
        close_prices = torch.full((len(timestamps), len(instruments)), float("nan"), dtype=torch.float64, device=device)
        availability_mask = torch.zeros((len(timestamps), len(instruments)), dtype=torch.bool, device=device)

        instrument_timestamp_lists: dict[str, list[int]] = {instrument_id: [] for instrument_id in instruments}
        instrument_timestamp_value_lists: dict[str, list[int]] = {instrument_id: [] for instrument_id in instruments}
        for bar in market_frame.bars:
            time_idx = timestamp_index[bar.timestamp]
            instrument_idx = instrument_index[bar.instrument_id]
            open_prices[time_idx, instrument_idx] = float(bar.open)
            close_prices[time_idx, instrument_idx] = float(bar.close)
            availability_mask[time_idx, instrument_idx] = True
            instrument_timestamp_lists[bar.instrument_id].append(time_idx)
            instrument_timestamp_value_lists[bar.instrument_id].append(_timestamp_to_sortable_int(bar.timestamp))

        return cls(
            device=device,
            timestamps=timestamps,
            instruments=instruments,
            timestamp_values=torch.tensor(
                [_timestamp_to_sortable_int(timestamp) for timestamp in timestamps],
                dtype=torch.int64,
                device=device,
            ),
            open_prices=open_prices,
            close_prices=close_prices,
            availability_mask=availability_mask,
            instrument_index=instrument_index,
            timestamp_index=timestamp_index,
            instrument_timestamp_values={
                instrument_id: torch.tensor(values, dtype=torch.int64, device=device)
                for instrument_id, values in instrument_timestamp_value_lists.items()
            },
            instrument_timestamp_indices={
                instrument_id: torch.tensor(indices, dtype=torch.int64, device=device)
                for instrument_id, indices in instrument_timestamp_lists.items()
            },
        )
