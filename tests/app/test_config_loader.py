from copy import deepcopy

from app.config_loader import load_active_inference_universe
from scripts.runtime_config import ConfigManager


PROFILE_PATH = "config/runtime_profiles/Gen6_1__group__bbva_peer_banks.yaml"


def _load_profile_config() -> dict:
    return deepcopy(ConfigManager(PROFILE_PATH).config)


def test_load_active_inference_universe_prefers_final_tickers_from_active_profile():
    config = _load_profile_config()

    universe = load_active_inference_universe(config)

    assert universe["source"] == "training_run.final_tickers_used"
    assert universe["anchor_ticker"] == "BBVA.MC"
    assert universe["peer_tickers"] == [
        "BKT.MC",
        "BNP.PA",
        "CABK.MC",
        "GLE.PA",
        "INGA.AS",
        "ISP.MI",
        "SAB.MC",
        "SAN.MC",
        "UCG.MI",
    ]
    assert "7203.T" not in universe["all_known_tickers"]
    assert "9984.T" not in universe["all_known_tickers"]


def test_load_active_inference_universe_falls_back_to_requested_tickers_when_final_is_empty():
    config = _load_profile_config()
    config["training_run"]["final_tickers_used"] = []

    universe = load_active_inference_universe(config)

    assert universe["source"] == "training_run.requested_tickers"
    assert universe["anchor_ticker"] == "BBVA.MC"
    assert universe["peer_tickers"] == [
        "SAN.MC",
        "CABK.MC",
        "SAB.MC",
        "BKT.MC",
        "BNP.PA",
        "GLE.PA",
        "ISP.MI",
        "UCG.MI",
        "INGA.AS",
    ]


def test_load_active_inference_universe_uses_first_ticker_when_anchor_is_missing():
    config = _load_profile_config()
    config["training_run"]["anchor_ticker"] = None
    config["training_run"]["predefined_group_name"] = None
    config["training_universe"]["anchor_ticker"] = None
    config["training_universe"]["predefined_group_name"] = None

    universe = load_active_inference_universe(config)

    assert universe["anchor_ticker"] == "BBVA.MC"
    assert universe["warnings"]
