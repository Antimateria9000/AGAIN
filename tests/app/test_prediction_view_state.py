from copy import deepcopy
import os
from pathlib import Path

from app.app import (
    _build_inference_cache_key,
    _resolve_effective_ticker,
    _should_render_persisted_result,
)
from scripts.runtime_config import ConfigManager


PROFILE_PATH = "config/runtime_profiles/Gen6_1__group__bbva_peer_banks.yaml"


def _load_profile_config() -> dict:
    return deepcopy(ConfigManager(PROFILE_PATH).config)


def test_resolve_effective_ticker_prioritizes_manual_then_peer_then_anchor():
    assert _resolve_effective_ticker(
        anchor_ticker="BBVA.MC",
        selected_peer="SAN.MC",
        manual_ticker=" msft ",
        use_manual=True,
    ) == "MSFT"
    assert _resolve_effective_ticker(
        anchor_ticker="BBVA.MC",
        selected_peer="SAN.MC",
        manual_ticker="",
        use_manual=False,
    ) == "SAN.MC"
    assert _resolve_effective_ticker(
        anchor_ticker="BBVA.MC",
        selected_peer="",
        manual_ticker="",
        use_manual=False,
    ) == "BBVA.MC"


def test_build_inference_cache_key_changes_when_profile_ticker_or_artifact_changes(tmp_path):
    config = _load_profile_config()
    model_path = tmp_path / "model.pth"
    meta_path = Path(f"{model_path}.meta.json")
    dataset_path = tmp_path / "processed_dataset.pt"
    model_path.write_bytes(b"checkpoint-v1")
    meta_path.write_text("{}", encoding="utf-8")
    dataset_path.write_bytes(b"dataset-v1")

    config["paths"]["model_save_path"] = str(model_path)
    config["data"]["processed_data_path"] = str(dataset_path)

    base_key = _build_inference_cache_key(
        config,
        selected_config_path="profile-a",
        effective_ticker="BBVA.MC",
        view_name="future_prediction",
    )
    ticker_key = _build_inference_cache_key(
        config,
        selected_config_path="profile-a",
        effective_ticker="SAN.MC",
        view_name="future_prediction",
    )
    profile_key = _build_inference_cache_key(
        config,
        selected_config_path="profile-b",
        effective_ticker="BBVA.MC",
        view_name="future_prediction",
    )

    os.utime(model_path, ns=(model_path.stat().st_atime_ns, model_path.stat().st_mtime_ns + 10_000))
    artifact_key = _build_inference_cache_key(
        config,
        selected_config_path="profile-a",
        effective_ticker="BBVA.MC",
        view_name="future_prediction",
    )

    assert base_key != ticker_key
    assert base_key != profile_key
    assert base_key != artifact_key


def test_should_render_persisted_result_only_when_cache_key_matches():
    expected_key = ("future_prediction", "profile-a", "BBVA.MC")

    assert _should_render_persisted_result(
        {"cache_key": expected_key, "result_payload": {"effective_ticker": "BBVA.MC"}},
        expected_key,
    )
    assert not _should_render_persisted_result(
        {"cache_key": ("future_prediction", "profile-a", "SAN.MC"), "result_payload": {"effective_ticker": "SAN.MC"}},
        expected_key,
    )
    assert not _should_render_persisted_result(
        {"cache_key": expected_key, "result_payload": None},
        expected_key,
    )
