"""Home Assistant entity tests for atomic battery-power acquisition evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from custom_components.anker_solix_official.acquisition import (
    BATTERY_POWER_REGISTER,
    AcquisitionFailureClass,
    BatteryPowerAcquisition,
)
from custom_components.anker_solix_official.sensor import (
    BatteryPowerAcquisitionSensor,
    ModbusLocalDeviceSensor,
)

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


def _data(value: int) -> dict[str, int]:
    return {
        "battery_charging_power": value,
        "battery_discharging_power": value,
    }


class _FakeCoordinator:
    def __init__(self, acquisition: BatteryPowerAcquisition) -> None:
        self.entry = type("Entry", (), {"entry_id": "test-entry"})()
        self.device_info = {"model": "Solarbank Max AC"}
        self.data: dict[str, Any] = {}
        self.battery_power_acquisition = acquisition

    def async_add_listener(self, listener):
        return lambda: None

    def is_connected(self) -> bool:
        return True

    def is_register_available(self, address: int) -> bool:
        return True

    def get_protected_value(self, entity_key: str) -> tuple[bool, Any]:
        return False, None


def test_atomic_sensor_exposes_valid_zero_and_same_generation_attributes() -> None:
    acquired_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    acquisition = BatteryPowerAcquisition(
        battery_power_raw_w=0,
        acquisition_valid=True,
        poll_generation=7,
        acquired_at=acquired_at,
        failure_class=AcquisitionFailureClass.NONE,
    )
    sensor = BatteryPowerAcquisitionSensor(_FakeCoordinator(acquisition))

    assert sensor.available is True
    assert sensor.native_value == 0
    assert sensor.extra_state_attributes == {
        "battery_power_raw_w": 0,
        "acquisition_valid": True,
        "poll_generation": 7,
        "acquired_at": acquired_at.isoformat(),
        "failure_class": "NONE",
        "modbus_address": BATTERY_POWER_REGISTER,
        "signed_semantics": "negative_charging_positive_discharging",
    }


def test_atomic_sensor_is_unavailable_on_failure_never_zero() -> None:
    acquisition = BatteryPowerAcquisition(
        battery_power_raw_w=None,
        acquisition_valid=False,
        poll_generation=8,
        acquired_at=None,
        failure_class=AcquisitionFailureClass.PARTIAL_READ_MISSING,
    )
    sensor = BatteryPowerAcquisitionSensor(_FakeCoordinator(acquisition))

    assert sensor.available is False
    assert sensor.native_value is None
    assert sensor.extra_state_attributes["failure_class"] == "PARTIAL_READ_MISSING"


def test_sensor_state_and_attributes_switch_together_on_coordinator_update() -> None:
    first_at = datetime(2026, 9, 10, 10, 0, tzinfo=UTC)
    coordinator = _FakeCoordinator(
        BatteryPowerAcquisition(
            battery_power_raw_w=0,
            acquisition_valid=True,
            poll_generation=1,
            acquired_at=first_at,
            failure_class=AcquisitionFailureClass.NONE,
        )
    )
    sensor = BatteryPowerAcquisitionSensor(coordinator)
    coordinator.battery_power_acquisition = BatteryPowerAcquisition(
        battery_power_raw_w=None,
        acquisition_valid=False,
        poll_generation=2,
        acquired_at=None,
        failure_class=AcquisitionFailureClass.PARTIAL_READ_MISSING,
    )
    writes = []
    sensor.async_write_ha_state = lambda: writes.append(True)

    assert sensor.native_value == 0
    assert sensor.extra_state_attributes["poll_generation"] == 1

    sensor._handle_coordinator_update()

    assert writes == [True]
    assert sensor.available is False
    assert sensor.native_value is None
    assert sensor.extra_state_attributes["poll_generation"] == 2
    assert sensor.extra_state_attributes["failure_class"] == "PARTIAL_READ_MISSING"


@pytest.mark.parametrize(
    ("raw", "expected_charge", "expected_discharge"),
    [(-300, 300, 0), (400, 0, 400), (0, 0, 0)],
)
def test_existing_charge_discharge_split_sensors_are_unchanged(
    raw: int, expected_charge: int, expected_discharge: int
) -> None:
    acquisition = BatteryPowerAcquisition(
        battery_power_raw_w=raw,
        acquisition_valid=True,
        poll_generation=1,
        acquired_at=datetime(2026, 9, 10, 10, 0, tzinfo=UTC),
        failure_class=AcquisitionFailureClass.NONE,
    )
    coordinator = _FakeCoordinator(acquisition)
    coordinator.data = _data(raw)
    charge = ModbusLocalDeviceSensor(
        coordinator,
        "battery_charging_power",
        {
            **BATTERY_DATA_POINTS["battery_charging_power"],
            "power_split_mode": "negative_only",
        },
    )
    discharge = ModbusLocalDeviceSensor(
        coordinator,
        "battery_discharging_power",
        {
            **BATTERY_DATA_POINTS["battery_discharging_power"],
            "power_split_mode": "positive_only",
        },
    )

    assert charge.native_value == expected_charge
    assert discharge.native_value == expected_discharge
