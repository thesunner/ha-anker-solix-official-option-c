"""Acceptance tests for atomic signed battery-power acquisition evidence."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from custom_components.anker_solix_official.acquisition import (
    BATTERY_POWER_REGISTER,
    BATTERY_POWER_REGISTER_WORDS,
    AcquisitionFailureClass,
    BatteryPowerAcquisition,
    BatteryPowerReadOutcome,
    RegisterReadDiagnostics,
)
from custom_components.anker_solix_official.coordinator import (
    AnkerSolixOfficialCoordinator,
)
from tests.test_modbus_client import _FakePymodbusClient, _FakeResponse, _make_client

BATTERY_DATA_POINTS = {
    "battery_charging_power": {
        "address": BATTERY_POWER_REGISTER,
        "count": 2,
        "data_type": "INT32",
    },
    "battery_discharging_power": {
        "address": BATTERY_POWER_REGISTER,
        "count": 2,
        "data_type": "INT32",
    },
}
BATTERY_BATCH_RANGES = [(BATTERY_POWER_REGISTER, BATTERY_POWER_REGISTER + 1, "input")]


def _coordinator() -> AnkerSolixOfficialCoordinator:
    coordinator = object.__new__(AnkerSolixOfficialCoordinator)
    coordinator._device_config_cache = BATTERY_DATA_POINTS
    coordinator._battery_power_poll_generation = 0
    coordinator._battery_power_requires_fresh_sample = False
    coordinator._battery_power_acquisition = BatteryPowerAcquisition(
        battery_power_raw_w=None,
        acquisition_valid=False,
        poll_generation=0,
        acquired_at=None,
        failure_class=AcquisitionFailureClass.DISCONNECTED,
    )
    return coordinator


def _valid_outcome(
    value: int, acquired_at: datetime | None = None
) -> BatteryPowerReadOutcome:
    return BatteryPowerReadOutcome(
        successful_words=BATTERY_POWER_REGISTER_WORDS,
        failed_words=frozenset(),
        battery_power_raw_w=value,
        sample_acquired_at=acquired_at
        or datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
        failure_class=AcquisitionFailureClass.NONE,
    )


def _invalid_outcome(
    failure: AcquisitionFailureClass,
    *,
    successful_words: frozenset[int] = frozenset(),
) -> BatteryPowerReadOutcome:
    return BatteryPowerReadOutcome(
        successful_words=successful_words,
        failed_words=BATTERY_POWER_REGISTER_WORDS - successful_words,
        failure_class=failure,
    )


def _diagnostics(
    outcome: BatteryPowerReadOutcome,
    *,
    connected: bool = True,
) -> RegisterReadDiagnostics:
    return RegisterReadDiagnostics(connected=connected, battery_power=outcome)


def _complete(
    coordinator: AnkerSolixOfficialCoordinator,
    outcome: BatteryPowerReadOutcome,
    *,
    data: dict[str, Any] | None = None,
    connected: bool = True,
) -> BatteryPowerAcquisition:
    generation = coordinator._begin_battery_power_poll()
    return coordinator._complete_battery_power_acquisition(
        {"some_data": 1} if data is None else data,
        _diagnostics(outcome, connected=connected),
        poll_generation=generation,
    )


@pytest.mark.parametrize("value", [400, -300, 0])
def test_signed_value_and_genuine_zero_are_valid(value: int) -> None:
    coordinator = _coordinator()
    acquired_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)

    result = _complete(coordinator, _valid_outcome(value, acquired_at))

    assert result == BatteryPowerAcquisition(
        battery_power_raw_w=value,
        acquisition_valid=True,
        poll_generation=1,
        acquired_at=acquired_at,
        failure_class=AcquisitionFailureClass.NONE,
    )


def test_repeated_unchanged_zero_advances_generation_and_timestamp() -> None:
    coordinator = _coordinator()
    first_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    second_at = first_at + timedelta(seconds=5)

    first = _complete(coordinator, _valid_outcome(0, first_at))
    second = _complete(coordinator, _valid_outcome(0, second_at))

    assert (first.battery_power_raw_w, second.battery_power_raw_w) == (0, 0)
    assert (first.poll_generation, second.poll_generation) == (1, 2)
    assert (first.acquired_at, second.acquired_at) == (first_at, second_at)


@pytest.mark.parametrize(
    "failure",
    [
        AcquisitionFailureClass.RANGE_MISSING,
        AcquisitionFailureClass.PARTIAL_READ_MISSING,
        AcquisitionFailureClass.DECODE_FAILURE,
        AcquisitionFailureClass.MODBUS_ERROR,
    ],
)
def test_classified_failure_never_inherits_value_or_timestamp(
    failure: AcquisitionFailureClass,
) -> None:
    coordinator = _coordinator()
    _complete(coordinator, _valid_outcome(400))

    result = _complete(coordinator, _invalid_outcome(failure))

    assert result.poll_generation == 2
    assert result.acquisition_valid is False
    assert result.battery_power_raw_w is None
    assert result.acquired_at is None
    assert result.failure_class is failure


def test_failed_poll_advances_generation_exactly_once() -> None:
    coordinator = _coordinator()

    generation = coordinator._begin_battery_power_poll()
    provisional = coordinator.battery_power_acquisition
    result = coordinator._complete_battery_power_acquisition(
        {},
        _diagnostics(
            _invalid_outcome(AcquisitionFailureClass.MODBUS_ERROR),
            connected=True,
        ),
        poll_generation=generation,
    )

    assert generation == 1
    assert provisional.poll_generation == 1
    assert provisional.acquisition_valid is False
    assert result.poll_generation == 1
    assert result.failure_class is AcquisitionFailureClass.MODBUS_ERROR


def test_connected_poll_without_read_outcome_is_complete_poll_failure() -> None:
    coordinator = _coordinator()
    generation = coordinator._begin_battery_power_poll()

    result = coordinator._complete_battery_power_acquisition(
        {},
        RegisterReadDiagnostics(connected=True),
        poll_generation=generation,
    )

    assert result.poll_generation == generation == 1
    assert result.acquisition_valid is False
    assert result.failure_class is AcquisitionFailureClass.COMPLETE_POLL_FAILURE


def test_disconnected_poll_is_invalid_without_a_second_generation() -> None:
    coordinator = _coordinator()
    generation = coordinator._begin_battery_power_poll()

    result = coordinator._complete_battery_power_acquisition(
        {},
        _diagnostics(
            _invalid_outcome(AcquisitionFailureClass.MODBUS_ERROR),
            connected=False,
        ),
        poll_generation=generation,
    )

    assert result.poll_generation == generation == 1
    assert result.failure_class is AcquisitionFailureClass.DISCONNECTED
    assert result.acquisition_valid is False


def test_reconnect_stays_invalid_until_first_fresh_two_word_sample() -> None:
    coordinator = _coordinator()
    coordinator._battery_power_requires_fresh_sample = True

    before_fresh = _complete(
        coordinator,
        _invalid_outcome(
            AcquisitionFailureClass.PARTIAL_READ_MISSING,
            successful_words=frozenset({BATTERY_POWER_REGISTER}),
        ),
    )
    fresh_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    fresh = _complete(coordinator, _valid_outcome(-250, fresh_at))

    assert (
        before_fresh.failure_class is AcquisitionFailureClass.RECONNECT_NO_FRESH_SAMPLE
    )
    assert before_fresh.acquisition_valid is False
    assert fresh.acquisition_valid is True
    assert fresh.battery_power_raw_w == -250
    assert fresh.poll_generation == before_fresh.poll_generation + 1
    assert fresh.acquired_at == fresh_at
    assert coordinator._battery_power_requires_fresh_sample is False


async def test_two_word_full_batch_success_is_valid() -> None:
    acquired_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=[0, 400])],
    )
    client = _make_client(fake)
    client._utcnow = lambda: acquired_at

    data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert data == {
        "battery_charging_power": 400,
        "battery_discharging_power": 400,
    }
    assert outcome.acquisition_valid is True
    assert outcome.successful_words == BATTERY_POWER_REGISTER_WORDS
    assert outcome.battery_power_raw_w == 400
    assert outcome.sample_acquired_at == acquired_at


async def test_fallback_10008_and_10009_success_is_valid_at_later_receipt() -> None:
    first_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    second_at = first_at + timedelta(milliseconds=10)
    times = iter((first_at, second_at))
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(error=True),
            _FakeResponse(registers=[0]),
            _FakeResponse(registers=[400]),
        ],
    )
    client = _make_client(fake)
    client._utcnow = lambda: next(times)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.acquisition_valid is True
    assert outcome.successful_words == BATTERY_POWER_REGISTER_WORDS
    assert outcome.battery_power_raw_w == 400
    assert outcome.sample_acquired_at == second_at


async def test_fallback_10008_success_10009_failure_is_invalid() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(error=True),
            _FakeResponse(registers=[0]),
            _FakeResponse(error=True),
        ],
    )
    client = _make_client(fake)

    data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert data == {}
    assert outcome.acquisition_valid is False
    assert outcome.successful_words == frozenset({BATTERY_POWER_REGISTER})
    assert outcome.failed_words == frozenset({BATTERY_POWER_REGISTER + 1})
    assert outcome.failure_class is AcquisitionFailureClass.PARTIAL_READ_MISSING
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None


async def test_fallback_10008_failure_10009_success_is_invalid() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(error=True),
            _FakeResponse(error=True),
            _FakeResponse(registers=[400]),
        ],
    )
    client = _make_client(fake)

    data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert data == {}
    assert outcome.acquisition_valid is False
    assert outcome.successful_words == frozenset({BATTERY_POWER_REGISTER + 1})
    assert outcome.failed_words == frozenset({BATTERY_POWER_REGISTER})
    assert outcome.failure_class is AcquisitionFailureClass.PARTIAL_READ_MISSING
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None


async def test_adversarial_legacy_default_zero_is_strict_decode_failure() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=["malformed", 0])],
    )
    client = _make_client(fake)

    data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power
    coordinator = _coordinator()
    acquisition = _complete(coordinator, outcome, data=data)

    assert data == {
        "battery_charging_power": 0,
        "battery_discharging_power": 0,
    }
    assert outcome.acquisition_valid is False
    assert outcome.failure_class is AcquisitionFailureClass.DECODE_FAILURE
    assert outcome.battery_power_raw_w is None
    assert acquisition.acquisition_valid is False
    assert acquisition.battery_power_raw_w is None
    assert acquisition.failure_class is AcquisitionFailureClass.DECODE_FAILURE


async def test_unexpected_strict_decoder_exception_is_decode_failure() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=[0, 400])],
    )
    client = _make_client(fake)

    def _raise_unexpected(registers: list[Any]) -> int:
        raise RuntimeError("controlled strict decoder fault")

    client._decode_battery_power_int32_strict = _raise_unexpected

    data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert data == {
        "battery_charging_power": 400,
        "battery_discharging_power": 400,
    }
    assert outcome.acquisition_valid is False
    assert outcome.failure_class is AcquisitionFailureClass.DECODE_FAILURE
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None


async def test_early_sample_timestamp_is_not_advanced_by_late_unrelated_range() -> None:
    sample_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    later_range_finished = asyncio.Event()

    class _DelayedLaterRangeClient(_FakePymodbusClient):
        async def read_input_registers(self, address: int, count: int) -> Any:
            if address == 10100:
                await asyncio.sleep(0.02)
                later_range_finished.set()
            return await super().read_input_registers(address, count)

    early_range = [0] * 51
    early_range[8:10] = [0, 400]
    fake = _DelayedLaterRangeClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(registers=early_range),
            _FakeResponse(registers=[7]),
        ],
    )
    client = _make_client(fake)
    client._utcnow = lambda: sample_at
    data_points = {
        **BATTERY_DATA_POINTS,
        "later_value": {
            "address": 10100,
            "count": 1,
            "data_type": "UINT16",
        },
    }
    batch_ranges = [(10000, 10050, "input"), (10100, 10100, "input")]

    data = await client.get_all_data(data_points, batch_ranges)
    assert later_range_finished.is_set()
    outcome = client.get_last_read_diagnostics().battery_power
    coordinator = _coordinator()
    acquisition = _complete(coordinator, outcome, data=data)

    assert data["later_value"] == 7
    assert outcome.sample_acquired_at == sample_at
    assert acquisition.acquired_at == sample_at


async def test_success_then_partial_failure_cannot_repeat_valid_zero() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(registers=[0, 0]),
            _FakeResponse(error=True),
            _FakeResponse(registers=[0]),
            _FakeResponse(error=True),
        ],
    )
    client = _make_client(fake)
    coordinator = _coordinator()

    first_data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    first = _complete(
        coordinator, client.get_last_read_diagnostics().battery_power, data=first_data
    )
    second_data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    second = _complete(
        coordinator, client.get_last_read_diagnostics().battery_power, data=second_data
    )

    assert first.acquisition_valid is True
    assert first.battery_power_raw_w == 0
    assert second.poll_generation == first.poll_generation + 1
    assert second.acquisition_valid is False
    assert second.battery_power_raw_w is None
    assert second.failure_class is AcquisitionFailureClass.PARTIAL_READ_MISSING


async def test_failure_then_recovery_accepts_only_new_physical_sample() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(error=True),
            _FakeResponse(error=True),
            _FakeResponse(error=True),
            _FakeResponse(registers=[0xFFFF, 0xFF06]),
        ],
    )
    client = _make_client(fake)
    coordinator = _coordinator()
    coordinator._battery_power_requires_fresh_sample = True

    failed_data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    failed = _complete(
        coordinator, client.get_last_read_diagnostics().battery_power, data=failed_data
    )
    recovered_data = await client.get_all_data(
        BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES
    )
    recovered = _complete(
        coordinator,
        client.get_last_read_diagnostics().battery_power,
        data=recovered_data,
    )

    assert failed.acquisition_valid is False
    assert failed.failure_class is AcquisitionFailureClass.RECONNECT_NO_FRESH_SAMPLE
    assert recovered.poll_generation == failed.poll_generation + 1
    assert recovered.acquisition_valid is True
    assert recovered.battery_power_raw_w == -250


def test_integration_lifecycle_reset_requires_source_reestablishment() -> None:
    old_lifecycle = _coordinator()
    _complete(old_lifecycle, _valid_outcome(0))
    old_last = _complete(old_lifecycle, _valid_outcome(0))

    restarted = _coordinator()
    assert restarted.battery_power_acquisition.poll_generation == 0
    assert restarted.battery_power_acquisition.acquisition_valid is False
    assert restarted.battery_power_acquisition.failure_class is (
        AcquisitionFailureClass.DISCONNECTED
    )

    first_new = _complete(restarted, _valid_outcome(125))

    assert old_last.poll_generation == 2
    assert first_new.poll_generation == 1
    assert first_new.acquisition_valid is True
    assert first_new.battery_power_raw_w == 125
    # A consumer must establish a new source lifecycle/session; generation 1
    # must never be interpreted as following generation 2 from the old process.


async def test_timeout_is_classified_as_modbus_error() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        connect_result=True,
        read_input_registers_queue=[asyncio.TimeoutError(), asyncio.TimeoutError()],
    )
    client = _make_client(fake)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.failure_class is AcquisitionFailureClass.MODBUS_ERROR
    assert outcome.acquisition_valid is False


async def test_short_range_is_classified_as_range_missing() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=[0])],
    )
    client = _make_client(fake)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_BATCH_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.failure_class is AcquisitionFailureClass.RANGE_MISSING
    assert outcome.acquisition_valid is False
