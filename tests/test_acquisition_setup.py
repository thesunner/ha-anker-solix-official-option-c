"""Tests for additive sensor registration without changing existing entities."""

from custom_components.anker_solix_official.base_entity import (
    async_setup_entities_with_retry,
)


class _SetupCoordinator:
    def __init__(self, data_points):
        self.data_points = data_points
        self.ip_address = "127.0.0.1"

    async def ensure_config_ready(self):
        return self.data_points

    async def get_device_data_points(self):
        return self.data_points


async def test_additional_factory_appends_one_atomic_sensor() -> None:
    coordinator = _SetupCoordinator(
        {"existing": {"address": 10008, "data_type_category": "read"}}
    )
    added = []

    await async_setup_entities_with_retry(
        hass=None,
        coordinator=coordinator,
        async_add_entities=lambda entities: added.extend(entities),
        entity_filter=lambda key, config: True,
        entity_factory=lambda coordinator, key, config: ("existing", key),
        platform_name="sensor",
        additional_entity_factory=lambda data_points: [("atomic", 10008)],
    )

    assert added == [("existing", "existing"), ("atomic", 10008)]


async def test_existing_platform_calls_without_additional_factory_are_unchanged() -> (
    None
):
    coordinator = _SetupCoordinator(
        {"existing": {"address": 10014, "data_type_category": "read"}}
    )
    added = []

    await async_setup_entities_with_retry(
        hass=None,
        coordinator=coordinator,
        async_add_entities=lambda entities: added.extend(entities),
        entity_filter=lambda key, config: True,
        entity_factory=lambda coordinator, key, config: ("existing", key),
        platform_name="sensor",
    )

    assert added == [("existing", "existing")]
