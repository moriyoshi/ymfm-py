"""
ymfm - Python bindings for the ymfm FM synthesis library

This module provides Python bindings for ymfm, a collection of Yamaha FM sound
chip emulators supporting 47+ chip variants.

Basic usage:
    import ymfm

    # Create a chip instance with clock frequency
    chip = ymfm.YM2612(clock=7670453)

    # Reset and write to registers
    chip.reset()
    chip.write(0, 0x30)  # Select register
    chip.write(1, 0x7F)  # Write value

    # Generate audio samples
    samples = chip.generate(1024)  # Returns memoryview of int32 values
"""

from ymfm._core import (
    DS1001,
    Y8950,
    # SSG family
    YM2149,
    # OPM family
    YM2151,
    YM2164,
    # OPN family
    YM2203,
    YM2413,
    # OPZ family
    YM2414,
    YM2423,
    YM2608,
    YM2610,
    YM2610B,
    YM2612,
    YM3438,
    # OPL family
    YM3526,
    YM3533,
    # OPQ family
    YM3806,
    YM3812,
    YMF262,
    YMF276,
    YMF278B,
    YMF281,
    YMF288,
    YMF289B,
    # Enums
    AccessClass,
    # Interface classes
    ChipInterface,
    MemoryInterface,
    SsgOverride,
)
from ._types import (
    Chip,
    ChipOutput,
)
from ._version import __version__


__all__ = [
    "__version__",
    # Type aliases and protocols
    "Chip",
    "ChipOutput",
    # Enums
    "AccessClass",
    # Interface classes
    "ChipInterface",
    "MemoryInterface",
    "SsgOverride",
    # OPL family
    "YM3526",
    "Y8950",
    "YM3812",
    "YMF262",
    "YM2413",
    "YMF289B",
    "YMF278B",
    "YM2423",
    "YMF281",
    "DS1001",
    # OPN family
    "YM2203",
    "YM2608",
    "YM2610",
    "YM2610B",
    "YM2612",
    "YM3438",
    "YMF276",
    "YMF288",
    # OPM family
    "YM2151",
    "YM2164",
    # SSG family
    "YM2149",
    # OPZ family
    "YM2414",
    # OPQ family
    "YM3806",
    "YM3533",
]
