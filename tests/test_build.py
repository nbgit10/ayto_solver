"""Tests for reproducible static-data build settings."""

import random

from build import build_timestamp, configure_build_random_seed


def test_build_timestamp_uses_source_date_epoch(monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "0")

    assert build_timestamp() == "1970-01-01T00:00:00+00:00"


def test_build_random_seed_repeats_graph_search_order(monkeypatch):
    monkeypatch.setenv("AYTO_BUILD_RANDOM_SEED", "17")
    state = random.getstate()

    try:
        configure_build_random_seed()
        first = [random.random() for _ in range(5)]
        configure_build_random_seed()
        second = [random.random() for _ in range(5)]
    finally:
        random.setstate(state)

    assert first == second
