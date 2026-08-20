from src.data.medallion_stages import resolve_runtime_settings


def test_fast_mode_uses_smaller_synthetic_scale():
    settings = resolve_runtime_settings(fast_mode=True, force_refresh=False)

    assert settings["clean_data_directories"] is False
    assert settings["bronze"]["num_customers"] == 200
    assert settings["bronze"]["num_days"] == 3
    assert settings["bronze"]["target_transactions"] == 5_000


def test_full_mode_targets_one_million_transactions():
    settings = resolve_runtime_settings(fast_mode=False, force_refresh=False)

    assert settings["bronze"]["target_transactions"] == 1_000_000
    assert settings["anomaly"]["anomaly_ratio"] == 0.015


def test_force_refresh_enables_directory_cleanup():
    settings = resolve_runtime_settings(fast_mode=False, force_refresh=True)

    assert settings["clean_data_directories"] is True
