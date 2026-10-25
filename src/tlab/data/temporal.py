from __future__ import annotations

from dataclasses import dataclass

SECONDS_PER_DAY = 24 * 60 * 60


@dataclass(frozen=True)
class TemporalBoundaries:
    end_raw: float
    train_cutoff_raw: float
    validation_start_raw: float
    test_start_raw: float
    rm_validation_start_raw: float
    rl_anchor_start_raw: float


def make_boundaries(
    max_timestamp_raw: float,
    timestamp_unit_seconds: float,
    timestamp_bin_seconds: float,
    train_days: int = 21,
    validation_days: int = 14,
    test_days: int = 7,
    rm_fit_days: int = 7,
    rl_anchor_days: int = 7,
) -> TemporalBoundaries:
    """Build half-open global cutoffs in source timestamp units after unit verification."""
    if timestamp_unit_seconds <= 0 or timestamp_bin_seconds <= 0:
        raise ValueError("timestamp unit and bin duration must be positive")
    if test_days <= 0 or validation_days <= 0 or train_days <= 0:
        raise ValueError("temporal split windows must be positive")
    if train_days != validation_days + test_days:
        raise ValueError("train_days must equal validation_days + test_days (21 = 14 + 7 by default)")
    if rm_fit_days <= 0 or rl_anchor_days <= 0:
        raise ValueError("RM validation and RL anchor windows must be positive")

    end = max_timestamp_raw + timestamp_bin_seconds / timestamp_unit_seconds
    train_cutoff = end - train_days * SECONDS_PER_DAY / timestamp_unit_seconds
    validation_start = train_cutoff
    test_start = end - test_days * SECONDS_PER_DAY / timestamp_unit_seconds
    # The early remainder of Train is F_RM. The final two training weeks are
    # split into V_RM and A_RL; rm_fit_days is not a fixed-duration block.
    rm_validation_start = train_cutoff - (rm_fit_days + rl_anchor_days) * SECONDS_PER_DAY / timestamp_unit_seconds
    rl_anchor_start = train_cutoff - rl_anchor_days * SECONDS_PER_DAY / timestamp_unit_seconds
    return TemporalBoundaries(
        end_raw=end,
        train_cutoff_raw=train_cutoff,
        validation_start_raw=validation_start,
        test_start_raw=test_start,
        rm_validation_start_raw=rm_validation_start,
        rl_anchor_start_raw=rl_anchor_start,
    )
