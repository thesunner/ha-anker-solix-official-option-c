"""Atomic acquisition evidence for safety-relevant Modbus registers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

BATTERY_POWER_REGISTER = 10008
BATTERY_POWER_REGISTER_WORDS = frozenset(
    {BATTERY_POWER_REGISTER, BATTERY_POWER_REGISTER + 1}
)


class AcquisitionFailureClass(StrEnum):
    """Closed failure taxonomy for one completed register acquisition."""

    NONE = "NONE"
    RANGE_MISSING = "RANGE_MISSING"
    PARTIAL_READ_MISSING = "PARTIAL_READ_MISSING"
    DECODE_FAILURE = "DECODE_FAILURE"
    MODBUS_ERROR = "MODBUS_ERROR"
    COMPLETE_POLL_FAILURE = "COMPLETE_POLL_FAILURE"
    DISCONNECTED = "DISCONNECTED"
    RECONNECT_NO_FRESH_SAMPLE = "RECONNECT_NO_FRESH_SAMPLE"


@dataclass(frozen=True, slots=True)
class BatteryPowerReadOutcome:
    """Strict result for the two Modbus words backing register 10008."""

    successful_words: frozenset[int] = frozenset()
    failed_words: frozenset[int] = BATTERY_POWER_REGISTER_WORDS
    battery_power_raw_w: int | None = None
    sample_acquired_at: datetime | None = None
    failure_class: AcquisitionFailureClass = AcquisitionFailureClass.RANGE_MISSING

    @property
    def acquisition_valid(self) -> bool:
        """Return whether both words were received and strictly decoded."""
        return (
            self.successful_words == BATTERY_POWER_REGISTER_WORDS
            and not self.failed_words
            and isinstance(self.battery_power_raw_w, int)
            and not isinstance(self.battery_power_raw_w, bool)
            and self.sample_acquired_at is not None
            and self.failure_class is AcquisitionFailureClass.NONE
        )

    def __post_init__(self) -> None:
        """Reject outcomes that could give an invalid read fresh semantics."""
        if not self.successful_words <= BATTERY_POWER_REGISTER_WORDS:
            raise ValueError("successful_words contains an unrelated register")
        if not self.failed_words <= BATTERY_POWER_REGISTER_WORDS:
            raise ValueError("failed_words contains an unrelated register")
        if self.successful_words & self.failed_words:
            raise ValueError("a component word cannot both succeed and fail")
        if self.acquisition_valid:
            assert self.sample_acquired_at is not None
            if self.sample_acquired_at.tzinfo is None or (
                self.sample_acquired_at.utcoffset() is None
            ):
                raise ValueError("sample_acquired_at must be timezone-aware")
            if self.sample_acquired_at.utcoffset() != UTC.utcoffset(
                self.sample_acquired_at
            ):
                raise ValueError("sample_acquired_at must be UTC")
        elif (
            self.battery_power_raw_w is not None
            or self.sample_acquired_at is not None
            or self.failure_class is AcquisitionFailureClass.NONE
        ):
            raise ValueError("invalid read outcome must not expose value or timestamp")


@dataclass(frozen=True, slots=True)
class RegisterReadDiagnostics:
    """Immutable register outcomes captured from one Modbus read cycle."""

    successful_registers: frozenset[int] = frozenset()
    failed_registers: frozenset[int] = frozenset()
    failure_classes: tuple[tuple[int, AcquisitionFailureClass], ...] = ()
    connected: bool = False
    battery_power: BatteryPowerReadOutcome = BatteryPowerReadOutcome()

    def failure_for(self, address: int) -> AcquisitionFailureClass | None:
        """Return the classified root failure for one register, if present."""
        return dict(self.failure_classes).get(address)


@dataclass(frozen=True, slots=True)
class BatteryPowerAcquisition:
    """One coherent Home Assistant publication for signed register 10008."""

    battery_power_raw_w: int | None
    acquisition_valid: bool
    poll_generation: int
    acquired_at: datetime | None
    failure_class: AcquisitionFailureClass

    def __post_init__(self) -> None:
        """Reject internally inconsistent acquisition records."""
        if self.poll_generation < 0:
            raise ValueError("poll_generation must be non-negative")
        if self.acquisition_valid:
            if isinstance(self.battery_power_raw_w, bool) or not isinstance(
                self.battery_power_raw_w, int
            ):
                raise ValueError("valid acquisition requires an integer raw value")
            if self.acquired_at is None:
                raise ValueError("valid acquisition requires acquired_at")
            if (
                not isinstance(self.acquired_at, datetime)
                or self.acquired_at.tzinfo is None
                or self.acquired_at.utcoffset() is None
            ):
                raise ValueError("acquired_at must be timezone-aware")
            if self.acquired_at.utcoffset() != UTC.utcoffset(self.acquired_at):
                raise ValueError("acquired_at must be UTC")
            if self.failure_class is not AcquisitionFailureClass.NONE:
                raise ValueError("valid acquisition requires failure_class NONE")
        elif (
            self.battery_power_raw_w is not None
            or self.acquired_at is not None
            or self.failure_class is AcquisitionFailureClass.NONE
        ):
            raise ValueError("invalid acquisition must not expose a value or timestamp")
