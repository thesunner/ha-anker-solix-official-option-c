"""Atomic producer evidence for the signed battery-power Modbus response."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

BATTERY_POWER_REGISTER = 10008
BATTERY_POWER_REGISTER_WORDS = frozenset(
    {BATTERY_POWER_REGISTER, BATTERY_POWER_REGISTER + 1}
)

CONTROLLED_FORK_VERSION = "1.5.0-powermind-option-c.1"
UPSTREAM_BASE_COMMIT = "3ecc42b9cb30beef24377f47486cb1eee985e1e3"
PRODUCER_REVISION = "option-c-register-10008-v2"
CAPABILITY_REVISION = "battery-power-int32-same-response-v1"
SIGNED_SEMANTICS = "negative_charging_positive_discharging"


class AcquisitionFailureClass(StrEnum):
    """Closed failure taxonomy for one producer attempt."""

    NONE = "NONE"
    RANGE_MISSING = "RANGE_MISSING"
    PARTIAL_READ_MISSING = "PARTIAL_READ_MISSING"
    DECODE_FAILURE = "DECODE_FAILURE"
    MODBUS_ERROR = "MODBUS_ERROR"
    COMPLETE_POLL_FAILURE = "COMPLETE_POLL_FAILURE"
    DISCONNECTED = "DISCONNECTED"
    RECONNECT_NO_FRESH_SAMPLE = "RECONNECT_NO_FRESH_SAMPLE"


def _require_utc(value: datetime, field_name: str) -> None:
    """Require a timezone-aware UTC timestamp."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    if value.utcoffset() != UTC.utcoffset(value):
        raise ValueError(f"{field_name} must be UTC")


@dataclass(frozen=True, slots=True)
class BatteryPowerReadOutcome:
    """Strict evidence captured from one physical Modbus response."""

    successful_words: frozenset[int] = frozenset()
    failed_words: frozenset[int] = BATTERY_POWER_REGISTER_WORDS
    battery_power_raw_w: int | None = None
    sample_acquired_at: datetime | None = None
    failure_class: AcquisitionFailureClass = AcquisitionFailureClass.RANGE_MISSING

    @property
    def acquisition_valid(self) -> bool:
        """Return whether one response supplied and decoded both words."""
        return (
            self.successful_words == BATTERY_POWER_REGISTER_WORDS
            and not self.failed_words
            and isinstance(self.battery_power_raw_w, int)
            and not isinstance(self.battery_power_raw_w, bool)
            and self.sample_acquired_at is not None
            and self.failure_class is AcquisitionFailureClass.NONE
        )

    def __post_init__(self) -> None:
        """Reject outcomes that could give invalid reads fresh semantics."""
        if not self.successful_words <= BATTERY_POWER_REGISTER_WORDS:
            raise ValueError("successful_words contains an unrelated register")
        if not self.failed_words <= BATTERY_POWER_REGISTER_WORDS:
            raise ValueError("failed_words contains an unrelated register")
        if self.successful_words & self.failed_words:
            raise ValueError("a component word cannot both succeed and fail")
        if self.successful_words | self.failed_words != BATTERY_POWER_REGISTER_WORDS:
            raise ValueError("every component word must be classified")

        if self.acquisition_valid:
            assert self.sample_acquired_at is not None
            _require_utc(self.sample_acquired_at, "sample_acquired_at")
        elif (
            self.battery_power_raw_w is not None
            or self.sample_acquired_at is not None
            or self.failure_class is AcquisitionFailureClass.NONE
        ):
            raise ValueError("invalid read outcome must not expose value or timestamp")


@dataclass(frozen=True, slots=True)
class RegisterReadDiagnostics:
    """Immutable physical-read diagnostics from one Modbus sweep."""

    successful_registers: frozenset[int] = frozenset()
    failed_registers: frozenset[int] = frozenset()
    failure_classes: tuple[tuple[int, AcquisitionFailureClass], ...] = ()
    connected: bool = False
    battery_power: BatteryPowerReadOutcome = BatteryPowerReadOutcome()

    def failure_for(self, address: int) -> AcquisitionFailureClass | None:
        """Return the classified root failure for one register."""
        return dict(self.failure_classes).get(address)


@dataclass(frozen=True, slots=True)
class BatteryPowerAcquisition:
    """One lifecycle- and attempt-scoped producer frame."""

    lifecycle_epoch: str
    poll_generation: int
    retrieved_at: datetime
    battery_power_raw_w: int | None
    acquisition_valid: bool
    acquired_at: datetime | None
    failure_class: AcquisitionFailureClass
    producer_version: str = CONTROLLED_FORK_VERSION
    producer_revision: str = PRODUCER_REVISION
    capability_revision: str = CAPABILITY_REVISION
    modbus_address: int = BATTERY_POWER_REGISTER
    signed_semantics: str = SIGNED_SEMANTICS

    @property
    def generation(self) -> int:
        """Measurement V2 name for the legacy poll generation."""
        return self.poll_generation

    @property
    def frame_valid(self) -> bool:
        """Measurement V2 name for acquisition validity."""
        return self.acquisition_valid

    @property
    def physical_acquired_at(self) -> datetime | None:
        """Measurement V2 name for the physical response timestamp."""
        return self.acquired_at

    @property
    def attempt_identity(self) -> str:
        """Return the stable identity of this attempt within its lifecycle."""
        return f"{self.lifecycle_epoch}:{self.poll_generation}"

    def __post_init__(self) -> None:
        """Reject internally inconsistent producer frames."""
        if not self.lifecycle_epoch:
            raise ValueError("lifecycle_epoch must not be empty")
        if self.poll_generation < 0:
            raise ValueError("poll_generation must be non-negative")
        _require_utc(self.retrieved_at, "retrieved_at")
        if not self.producer_version or not self.producer_revision:
            raise ValueError("producer identity must not be empty")
        if not self.capability_revision:
            raise ValueError("capability_revision must not be empty")
        if self.modbus_address != BATTERY_POWER_REGISTER:
            raise ValueError("modbus_address must identify register 10008")
        if self.signed_semantics != SIGNED_SEMANTICS:
            raise ValueError("signed_semantics is not the producer contract")

        if self.acquisition_valid:
            if isinstance(self.battery_power_raw_w, bool) or not isinstance(
                self.battery_power_raw_w, int
            ):
                raise ValueError("valid acquisition requires an integer raw value")
            if self.acquired_at is None:
                raise ValueError("valid acquisition requires acquired_at")
            _require_utc(self.acquired_at, "acquired_at")
            if self.failure_class is not AcquisitionFailureClass.NONE:
                raise ValueError("valid acquisition requires failure_class NONE")
        elif (
            self.battery_power_raw_w is not None
            or self.acquired_at is not None
            or self.failure_class is AcquisitionFailureClass.NONE
        ):
            raise ValueError("invalid acquisition must not expose a value or timestamp")
