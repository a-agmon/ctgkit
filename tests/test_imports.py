"""Smoke test: the public surface imports cleanly."""


def test_package_imports():
    import ctgkit

    assert ctgkit.__version__


def test_public_api():
    from ctgkit import (
        analyze,
        analyze_service,
        ServiceConfig,
        RECOMMENDED,
        load_csv,
        from_arrays,
        Signal,
        EpochResult,
        Concern,
        Category,
        AlertLevel,
        Trend,
        GUIDELINE_PACKS,
        list_packs,
    )

    assert callable(analyze)
    assert callable(analyze_service)
    assert callable(list_packs)
