from __future__ import annotations

LISTEN_COLUMNS = (
    "uid", "item_id", "timestamp", "is_organic", "played_ratio_pct", "track_length_seconds"
)
EVENT_ID_COLUMNS = ("uid", "item_id", "timestamp")


def require_columns(columns: set[str], required: tuple[str, ...], table: str) -> None:
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")
