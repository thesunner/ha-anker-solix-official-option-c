"""v1.5 regressions for atomic register-10008 producer evidence."""

from __future__ import annotations

import asyncio
import importlib
import logging
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from custom_components.anker_solix_official import sensor as sensor_module
from custom_components.anker_solix_official.const import DOMAIN
from custom_components.anker_solix_official.coordinator import (
    AnkerSolixOfficialCoordinator,
)

from tests.test_modbus_client import _FakePymodbusClient, _FakeResponse, _make_client

BATTERY_REGISTER = 10008
BATTERY_WORDS = frozenset({10008, 10009})
BATTERY_DATA_POINTS = {
    "battery_charging_power": {
        "address": BATTERY_REGISTER,
        "count": 2,
        "data_type": "INT32",
        "unit": "W",
        "gain": 1,
        "power_split_mode": "negative_only",
    },
    "battery_discharging_power": {
        "address": BATTERY_REGISTER,
        "count": 2,
        "data_type": "INT32",
        "unit": "W",
        "gain": 1,
        "power_split_mode": "positive_only",
    },
}
BATTERY_RANGES = [(BATTERY_REGISTER, BATTERY_REGISTER + 1, "input")]
T0 = datetime(2026, 9, 22, 10, 0, tzinfo=UTC)


def _acquisition_module():
    """Load the wished-for producer contract and fail clearly on official v1.5."""
    try:
        return importlib.import_module(
            "custom_components.anker_solix_official.acquisition"
        )
    except ModuleNotFoundError:
        pytest.fail("official v1.5.0 has no acquisition producer contract")


def _valid_outcome(value: int, acquired_at: datetime = T0):
    acq = _acquisition_module()
    return acq.BatteryPowerReadOutcome(
        successful_words=BATTERY_WORDS,
        failed_words=frozenset(),
        battery_power_raw_w=value,
        sample_acquired_at=acquired_at,
        failure_class=acq.AcquisitionFailureClass.NONE,
    )


def _invalid_outcome(failure_name: str = "RANGE_MISSING"):
    acq = _acquisition_module()
    return acq.BatteryPowerReadOutcome(
        failure_class=acq.AcquisitionFailureClass[failure_name]
    )


def _diagnostics(outcome, *, connected: bool = True):
    acq = _acquisition_module()
    return acq.RegisterReadDiagnostics(
        connected=connected,
        battery_power=outcome,
    )


def _frame(
    *,
    epoch: str = "epoch-a",
    generation: int = 1,
    value: int | None = -850,
    acquired_at: datetime | None = T0,
    failure_name: str = "NONE",
    retrieved_at: datetime = T0,
):
    acq = _acquisition_module()
    valid = failure_name == "NONE"
    return acq.BatteryPowerAcquisition(
        lifecycle_epoch=epoch,
        poll_generation=generation,
        retrieved_at=retrieved_at,
        battery_power_raw_w=value if valid else None,
        acquisition_valid=valid,
        acquired_at=acquired_at if valid else None,
        failure_class=acq.AcquisitionFailureClass[failure_name],
    )


class _SequenceManager:
    def __init__(self, frames: list[tuple[dict[str, Any], Any]]) -> None:
        self._frames = list(frames)
        self._diagnostics = None

    async def get_all_data(self, *_args, **_kwargs) -> dict[str, Any]:
        data, self._diagnostics = self._frames.pop(0)
        return dict(data)

    def get_last_read_diagnostics(self):
        return self._diagnostics


def _coordinator(manager: _SequenceManager, *, epoch: str = "epoch-a"):
    coordinator = object.__new__(AnkerSolixOfficialCoordinator)
    coordinator._initialize_battery_power_acquisition(lifecycle_epoch=epoch)
    coordinator.modbus_manager = manager
    coordinator._device_config_cache = BATTERY_DATA_POINTS
    coordinator._batch_ranges_cache = BATTERY_RANGES
    coordinator._config_cache_valid = True
    coordinator._full_config_cache = {"product_info": {"default_name": "Test"}}
    coordinator._initial_mode_sent = True
    coordinator._latest_data = {}
    coordinator._empty_frames = 0
    coordinator._consecutive_failures = 0
    coordinator._ever_connected = False
    coordinator._unavailable_registers = set()
    coordinator.ip_address = "127.0.0.1"
    coordinator.logger = logging.getLogger(__name__)
    coordinator.device_info = {"model": "--"}
    coordinator._async_connect_and_ensure_config = AsyncMock()
    coordinator._update_unavailable_registers = AsyncMock()
    coordinator._handle_refresh_failure = AsyncMock()
    coordinator._auto_set_mode_on_connect = AsyncMock()
    coordinator._override_model_with_product_name = lambda _data: None
    coordinator._inject_version_gates = lambda _data: None
    coordinator._log_data_update = lambda *_args: None
    coordinator._update_device_registry_info = lambda: None
    return coordinator


def test_contract_identity_and_failure_taxonomy_are_stable() -> None:
    acq = _acquisition_module()
    frame = _frame()

    assert acq.CONTROLLED_FORK_VERSION == "1.5.0-powermind-option-c.1"
    assert acq.UPSTREAM_BASE_COMMIT == "3ecc42b9cb30beef24377f47486cb1eee985e1e3"
    assert acq.PRODUCER_REVISION
    assert acq.CAPABILITY_REVISION
    assert frame.producer_version == acq.CONTROLLED_FORK_VERSION
    assert frame.producer_revision == acq.PRODUCER_REVISION
    assert frame.capability_revision == acq.CAPABILITY_REVISION
    assert frame.lifecycle_epoch == "epoch-a"
    assert frame.attempt_identity == "epoch-a:1"
    assert frame.frame_valid is True
    assert {item.value for item in acq.AcquisitionFailureClass} == {
        "NONE",
        "RANGE_MISSING",
        "PARTIAL_READ_MISSING",
        "DECODE_FAILURE",
        "MODBUS_ERROR",
        "COMPLETE_POLL_FAILURE",
        "DISCONNECTED",
        "RECONNECT_NO_FRESH_SAMPLE",
    }


def test_invalid_contract_cannot_expose_stale_value_or_timestamp() -> None:
    acq = _acquisition_module()
    with pytest.raises(ValueError, match="must not expose"):
        acq.BatteryPowerAcquisition(
            lifecycle_epoch="epoch-a",
            poll_generation=2,
            retrieved_at=T0,
            battery_power_raw_w=-850,
            acquisition_valid=False,
            acquired_at=T0,
            failure_class=acq.AcquisitionFailureClass.MODBUS_ERROR,
        )


async def test_fresh_success_strictly_decodes_one_physical_response() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=[0xFFFF, 0xFCAE])],
    )
    client = _make_client(fake)
    client._utcnow = lambda: T0

    data = await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert data["battery_charging_power"] == -850
    assert outcome.acquisition_valid is True
    assert outcome.battery_power_raw_w == -850
    assert outcome.sample_acquired_at == T0


async def test_repeated_equal_physical_responses_remain_distinct_acquisitions() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(registers=[0xFFFF, 0xFCAE]),
            _FakeResponse(registers=[0xFFFF, 0xFCAE]),
        ],
    )
    client = _make_client(fake)
    times = iter((T0, T0 + timedelta(seconds=5)))
    client._utcnow = lambda: next(times)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    first = client.get_last_read_diagnostics().battery_power
    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    second = client.get_last_read_diagnostics().battery_power

    assert first.battery_power_raw_w == second.battery_power_raw_w == -850
    assert first.sample_acquired_at == T0
    assert second.sample_acquired_at == T0 + timedelta(seconds=5)


async def test_separate_fallback_responses_cannot_form_atomic_evidence() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(error=True),
            _FakeResponse(registers=[0xFFFF]),
            _FakeResponse(registers=[0xFCAE]),
        ],
    )
    client = _make_client(fake)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.acquisition_valid is False
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None
    assert outcome.failure_class.value == "PARTIAL_READ_MISSING"


async def test_partial_two_word_response_is_invalid() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=[0xFFFF])],
    )
    client = _make_client(fake)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.acquisition_valid is False
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None
    assert outcome.failure_class.value == "PARTIAL_READ_MISSING"


@pytest.mark.parametrize("words", [["bad", 0], [True, 0], [0x10000, 0]])
async def test_strict_decode_rejects_malformed_words(words: list[Any]) -> None:
    fake = _FakePymodbusClient(
        connected=True,
        read_input_registers_queue=[_FakeResponse(registers=words)],
    )
    client = _make_client(fake)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.acquisition_valid is False
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None
    assert outcome.failure_class.value == "DECODE_FAILURE"


async def test_timeout_is_modbus_error_without_stale_success() -> None:
    fake = _FakePymodbusClient(
        connected=True,
        connect_result=True,
        read_input_registers_queue=[TimeoutError(), TimeoutError()],
    )
    client = _make_client(fake)

    await client.get_all_data(BATTERY_DATA_POINTS, BATTERY_RANGES)
    outcome = client.get_last_read_diagnostics().battery_power

    assert outcome.acquisition_valid is False
    assert outcome.battery_power_raw_w is None
    assert outcome.sample_acquired_at is None
    assert outcome.failure_class.value == "MODBUS_ERROR"


async def test_late_unrelated_range_does_not_move_physical_timestamp() -> None:
    later_finished = asyncio.Event()

    class _DelayedClient(_FakePymodbusClient):
        async def read_input_registers(self, address: int, count: int) -> Any:
            if address == 10100:
                await asyncio.sleep(0.01)
                later_finished.set()
            return await super().read_input_registers(address, count)

    early = [0] * 11
    early[8:10] = [0xFFFF, 0xFCAE]
    fake = _DelayedClient(
        connected=True,
        read_input_registers_queue=[
            _FakeResponse(registers=early),
            _FakeResponse(registers=[7]),
        ],
    )
    client = _make_client(fake)
    client._utcnow = lambda: T0
    data_points = {
        **BATTERY_DATA_POINTS,
        "later_value": {"address": 10100, "count": 1, "data_type": "UINT16"},
    }

    data = await client.get_all_data(
        data_points,
        [(10000, 10010, "input"), (10100, 10100, "input")],
    )
    outcome = client.get_last_read_diagnostics().battery_power

    assert later_finished.is_set()
    assert data["later_value"] == 7
    assert outcome.sample_acquired_at == T0


async def test_empty_frame_after_success_keeps_presentation_not_evidence() -> None:
    manager = _SequenceManager(
        [
            (
                {
                    "battery_charging_power": -1090,
                    "battery_discharging_power": -1090,
                },
                _diagnostics(_valid_outcome(-1090)),
            ),
            ({}, _diagnostics(_invalid_outcome())),
        ]
    )
    coordinator = _coordinator(manager)

    first = await coordinator._async_update_data()
    second = await coordinator._async_update_data()
    producer = coordinator.battery_power_acquisition

    assert second == first
    assert second["battery_charging_power"] == -1090
    assert producer.poll_generation == 2
    assert producer.attempt_identity == "epoch-a:2"
    assert producer.acquisition_valid is False
    assert producer.battery_power_raw_w is None
    assert producer.acquired_at is None
    assert producer.failure_class.value == "COMPLETE_POLL_FAILURE"


async def test_two_tolerated_empty_frames_never_inherit_timestamp() -> None:
    manager = _SequenceManager(
        [
            (
                {
                    "battery_charging_power": -1090,
                    "battery_discharging_power": -1090,
                },
                _diagnostics(_valid_outcome(-1090)),
            ),
            ({}, _diagnostics(_invalid_outcome())),
            ({}, _diagnostics(_invalid_outcome())),
        ]
    )
    coordinator = _coordinator(manager)

    await coordinator._async_update_data()
    await coordinator._async_update_data()
    second_empty = coordinator.battery_power_acquisition
    presentation = await coordinator._async_update_data()
    third = coordinator.battery_power_acquisition

    assert presentation["battery_charging_power"] == -1090
    assert second_empty.acquired_at is None
    assert third.acquired_at is None
    assert (second_empty.poll_generation, third.poll_generation) == (2, 3)
    assert second_empty.attempt_identity != third.attempt_identity


async def test_failure_recovery_requires_new_physical_response() -> None:
    presentation = {
        "battery_charging_power": -850,
        "battery_discharging_power": -850,
    }
    manager = _SequenceManager(
        [
            (presentation, _diagnostics(_valid_outcome(-850))),
            ({}, _diagnostics(_invalid_outcome())),
            (
                presentation,
                _diagnostics(_invalid_outcome("PARTIAL_READ_MISSING")),
            ),
            (
                presentation,
                _diagnostics(_valid_outcome(-850, T0 + timedelta(seconds=15))),
            ),
        ]
    )
    coordinator = _coordinator(manager)

    await coordinator._async_update_data()
    await coordinator._async_update_data()
    coordinator._battery_power_requires_fresh_sample = True
    await coordinator._async_update_data()
    stale = coordinator.battery_power_acquisition
    await coordinator._async_update_data()
    recovered = coordinator.battery_power_acquisition

    assert stale.acquisition_valid is False
    assert stale.failure_class.value == "RECONNECT_NO_FRESH_SAMPLE"
    assert stale.battery_power_raw_w is None
    assert recovered.acquisition_valid is True
    assert recovered.poll_generation == stale.poll_generation + 1
    assert recovered.acquired_at == T0 + timedelta(seconds=15)


def test_new_coordinator_lifecycle_breaks_generation_continuity() -> None:
    first = _coordinator(_SequenceManager([]), epoch="epoch-a")
    restarted = _coordinator(_SequenceManager([]), epoch="epoch-b")

    assert first.battery_power_acquisition.poll_generation == 0
    assert restarted.battery_power_acquisition.poll_generation == 0
    assert first.battery_power_acquisition.acquisition_valid is False
    assert restarted.battery_power_acquisition.acquisition_valid is False
    assert first.battery_power_acquisition.lifecycle_epoch != (
        restarted.battery_power_acquisition.lifecycle_epoch
    )
    assert first.battery_power_acquisition.attempt_identity != (
        restarted.battery_power_acquisition.attempt_identity
    )


class _SensorCoordinator:
    def __init__(self, acquisition) -> None:
        self.entry = SimpleNamespace(entry_id="test-entry")
        self.device_info = {"model": "Solarbank Max AC"}
        self.data = {
            "battery_charging_power": -850,
            "battery_discharging_power": -850,
        }
        self.last_update_success = True
        self.battery_power_acquisition = acquisition
        self.ip_address = "127.0.0.1"

    async def get_device_data_points(self):
        return BATTERY_DATA_POINTS

    def async_add_listener(self, _listener):
        return lambda: None

    def is_register_available(self, _address: int) -> bool:
        return True

    def get_protected_value(self, _key: str) -> tuple[bool, Any]:
        return False, None


async def test_acquisition_entity_is_actively_created_with_identity_attributes() -> None:
    frame = _frame()
    coordinator = _SensorCoordinator(frame)
    hass = SimpleNamespace(data={DOMAIN: {"entry": coordinator}})
    added: list[Any] = []

    await sensor_module.async_setup_entry(
        hass,
        SimpleNamespace(entry_id="entry"),
        added.extend,
    )

    producer_type = sensor_module.BatteryPowerAcquisitionSensor
    producers = [entity for entity in added if isinstance(entity, producer_type)]
    assert len(producers) == 1
    attrs = producers[0].extra_state_attributes
    assert attrs["battery_power_raw_w"] == -850
    assert attrs["acquisition_valid"] is True
    assert attrs["poll_generation"] == 1
    assert attrs["acquired_at"] == T0.isoformat()
    assert attrs["failure_class"] == "NONE"
    assert attrs["modbus_address"] == BATTERY_REGISTER
    assert attrs["signed_semantics"] == "negative_charging_positive_discharging"
    assert attrs["producer_version"] == "1.5.0-powermind-option-c.1"
    assert attrs["producer_revision"]
    assert attrs["capability_revision"]
    assert attrs["lifecycle_epoch"] == "epoch-a"
    assert attrs["attempt_identity"] == "epoch-a:1"
    assert attrs["retrieved_at"] == T0.isoformat()


@pytest.mark.parametrize(
    ("raw", "expected_charge", "expected_discharge"),
    [(-300, 300, 0), (400, 0, 400), (0, 0, 0)],
)
def test_existing_charge_discharge_sensor_semantics_are_unchanged(
    raw: int,
    expected_charge: int,
    expected_discharge: int,
) -> None:
    coordinator = _SensorCoordinator(_frame(value=raw))
    coordinator.data = {
        "battery_charging_power": raw,
        "battery_discharging_power": raw,
    }
    charge = sensor_module.ModbusLocalDeviceSensor(
        coordinator,
        "battery_charging_power",
        BATTERY_DATA_POINTS["battery_charging_power"],
    )
    discharge = sensor_module.ModbusLocalDeviceSensor(
        coordinator,
        "battery_discharging_power",
        BATTERY_DATA_POINTS["battery_discharging_power"],
    )

    assert charge.native_value == expected_charge
    assert discharge.native_value == expected_discharge
