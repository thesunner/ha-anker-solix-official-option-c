"""Regression for setup failure before a new physical producer sweep starts."""

from unittest.mock import AsyncMock

import pytest
from homeassistant.helpers.update_coordinator import UpdateFailed

from tests.test_acquisition_v15 import (
    _coordinator,
    _diagnostics,
    _SequenceManager,
    _valid_outcome,
)


async def test_pre_read_failure_cannot_reuse_previous_success_diagnostics() -> None:
    manager = _SequenceManager(
        [
            (
                {
                    "battery_charging_power": -850,
                    "battery_discharging_power": -850,
                },
                _diagnostics(_valid_outcome(-850)),
            )
        ]
    )
    coordinator = _coordinator(manager)
    await coordinator._async_update_data()
    coordinator._async_connect_and_ensure_config = AsyncMock(
        side_effect=UpdateFailed("cannot connect")
    )

    with pytest.raises(UpdateFailed):
        await coordinator._async_update_data()

    producer = coordinator.battery_power_acquisition
    assert producer.poll_generation == 2
    assert producer.acquisition_valid is False
    assert producer.battery_power_raw_w is None
    assert producer.acquired_at is None
    assert producer.failure_class.value == "DISCONNECTED"
