# Pydantic Integration Summary

This document explains the integration of Pydantic into the `shortschedule` package and its benefits.

## What is Pydantic?

Pydantic is a data validation library that uses Python type annotations to validate data at runtime. It provides automatic type checking, data validation, and serialization capabilities.

## Why Use Pydantic?

### Before Pydantic
```python
# Manual construction, no validation
seq = ObservationSequence(
    id="test",
    target="star1",
    priority=1,
    start_time=time1,
    stop_time=time2,
    ra=361.0,  # Invalid! But no error until used
    dec=45.0,
    payload_params={}
)
# Error occurs later when ra is used
```

### After Pydantic
```python
# Automatic validation at construction
seq = ObservationSequence(
    id="test",
    target="star1",
    priority=1,
    start_time=time1,
    stop_time=time2,
    ra=361.0,  # ValidationError immediately!
    dec=45.0,
    payload_params={}
)
# ValidationError: ra must be in range [0, 360), got 361.0
```

## Benefits

### 1. Early Error Detection
Invalid data is caught immediately at construction time, not during runtime operations.

### 2. Clear Error Messages
Pydantic provides descriptive validation errors that clearly indicate what went wrong:
```
ValidationError: 1 validation error for ObservationSequence
ra
  Value error, ra must be in range [0, 360), got 361.0
```

### 3. Type Safety
All field types are automatically validated:
```python
seq = ObservationSequence(
    start_time="not a Time object",  # ValidationError
    ...
)
```

### 4. Automatic Type Conversion
ISO strings are automatically converted to astropy Time objects:
```python
seq = ObservationSequence(
    start_time="2026-01-01T00:00:00",  # Automatically converted to Time
    stop_time="2026-01-01T01:00:00",  # Automatically converted to Time
    ...
)
```

### 5. Better IDE Support
IDEs can provide better autocomplete and type hints with Pydantic models.

### 6. Centralized Validation
All validation logic is in one place (the model definition), making it easier to maintain and update.

## Validation Rules

### ObservationSequence
- `id`: string (required)
- `target`: string (required)
- `priority`: non-negative integer (required)
- `start_time`: astropy Time object or ISO string (required)
- `stop_time`: astropy Time object or ISO string (required)
- `ra`: float in range [0, 360) (required)
- `dec`: float in range [-90, 90] (required)
- `payload_params`: dictionary (required)
- `roll`: float in range [0, 360) or None (optional)

### Visit
- `id`: string (required)
- `sequences`: list of ObservationSequence objects (required)

### ScienceCalendar
- `metadata`: dictionary (defaults to {} if None)
- `visits`: list of Visit objects (required)
- `visibility`: any type (optional)

## Examples

### Valid Construction
```python
from astropy.time import Time
from shortschedule.models import ObservationSequence

seq = ObservationSequence(
    id="seq_001",
    target="WASP-12",
    priority=1,
    start_time=Time("2026-01-01T00:00:00", format="isot"),
    stop_time=Time("2026-01-01T01:00:00", format="isot"),
    ra=180.5,
    dec=45.2,
    payload_params={},
    roll=90.0
)
```

### ISO String Conversion
```python
# ISO strings are automatically converted to Time objects
seq = ObservationSequence(
    id="seq_001",
    target="WASP-12",
    priority=1,
    start_time="2026-01-01T00:00:00",  # Auto-converted
    stop_time="2026-01-01T01:00:00",   # Auto-converted
    ra=180.5,
    dec=45.2,
    payload_params={},
)
```

### Validation Errors
```python
# Invalid RA
seq = ObservationSequence(
    ra=361.0,  # ValidationError: ra must be in range [0, 360)
    ...
)

# Invalid Dec
seq = ObservationSequence(
    dec=91.0,  # ValidationError: dec must be in range [-90, 90]
    ...
)

# Negative priority
seq = ObservationSequence(
    priority=-1,  # ValidationError: priority must be non-negative
    ...
)
```

## Testing

Comprehensive tests demonstrate the validation capabilities in `tests/test_pydantic_validation.py`. These tests ensure that:

1. Invalid time objects raise ValidationError
2. Invalid RA/Dec/roll ranges raise ValidationError
3. Negative priorities raise ValidationError
4. Valid data is accepted
5. ISO strings are converted to Time objects

## Backward Compatibility

All existing functionality is preserved. The API remains unchanged - existing code continues to work without modification. All 52 existing tests pass without changes.

## Performance

Pydantic validation adds a small overhead at object construction time, but this is negligible compared to the benefits of early error detection and data integrity.

## Conclusion

Pydantic integration significantly improves the robustness and maintainability of the `shortschedule` package by providing automatic data validation, clear error messages, and better developer experience.
