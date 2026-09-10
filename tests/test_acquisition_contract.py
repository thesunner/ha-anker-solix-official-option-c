"""Invariant tests for atomic register-10008 evidence value objects."""

from datetime import UTC, datetime, timedelta, timezone

import pytest

from custom_components.anker_solix_official.acquisition import (
    BATTERY_POWER_REGISTER,
    BATTERY_POWER_REGISTER_WORDS,
    AcquisitionFailureClass,
    BatteryPowerAcquisition,
    BatteryPowerReadOutcome,
)


def test_valid_contract_requires_timezone_aware_acquired_at() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        BatteryPowerAcquisition(
            battery_power_raw_w=0,
            acquisition_valid=True,
            poll_generation=1,
            acquired_at=datetime(2026, 9, 10, 10, 0),
            failure_class=AcquisitionFailureClass.NONE,
        )


def test_valid_contract_requires_utc_acquired_at() -> None:
    with pytest.raises(ValueError, match="must be UTC"):
        BatteryPowerAcquisition(
            battery_power_raw_w=0,
            acquisition_valid=True,
            poll_generation=1,
            acquired_at=datetime(
                2026, 9, 10, 10, 0, tzinfo=timezone(timedelta(hours=2))
            ),
            failure_class=AcquisitionFailureClass.NONE,
        )


def test_invalid_contract_rejects_stale_value_or_timestamp() -> None:
    with pytest.raises(ValueError, match="must not expose"):
        BatteryPowerAcquisition(
            battery_power_raw_w=0,
            acquisition_valid=False,
            poll_generation=2,
            acquired_at=datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
            failure_class=AcquisitionFailureClass.MODBUS_ERROR,
        )


def test_read_outcome_requires_both_component_words_for_validity() -> None:
    acquired_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    partial = BatteryPowerReadOutcome(
        successful_words=frozenset({BATTERY_POWER_REGISTER}),
        failed_words=frozenset({BATTERY_POWER_REGISTER + 1}),
        failure_class=AcquisitionFailureClass.PARTIAL_READ_MISSING,
    )
    complete = BatteryPowerReadOutcome(
        successful_words=BATTERY_POWER_REGISTER_WORDS,
        failed_words=frozenset(),
        battery_power_raw_w=0,
        sample_acquired_at=acquired_at,
        failure_class=AcquisitionFailureClass.NONE,
    )

    assert partial.acquisition_valid is False
    assert complete.acquisition_valid is True


def test_invalid_read_outcome_cannot_expose_value_or_timestamp() -> None:
    with pytest.raises(ValueError, match="must not expose"):
        BatteryPowerReadOutcome(
            successful_words=BATTERY_POWER_REGISTER_WORDS,
            failed_words=frozenset(),
            battery_power_raw_w=0,
            sample_acquired_at=datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
            failure_class=AcquisitionFailureClass.DECODE_FAILURE,
        )


def test_failure_taxonomy_contains_every_required_class() -> None:
    assert {failure.value for failure in AcquisitionFailureClass} == {
        "NONE",
        "RANGE_MISSING",
        "PARTIAL_READ_MISSING",
        "DECODE_FAILURE",
        "MODBUS_ERROR",
        "COMPLETE_POLL_FAILURE",
        "DISCONNECTED",
        "RECONNECT_NO_FRESH_SAMPLE",
    }
