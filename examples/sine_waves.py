#!/usr/bin/env python3
"""
Generate sine waves using all supported FM chips.

This example demonstrates how to program each chip type to generate
pure sine waves at specific frequencies (110Hz, 220Hz, 440Hz, 880Hz).

FM chips generate sine waves by default when using a single operator
with no modulation. The frequency is controlled by F-number/block (OPL/OPN)
or KC/KF (OPM) registers.
"""

import wave
from pathlib import Path
from typing import Callable, cast

import numpy as np

import ymfm

# Type alias for setup functions
SetupFunc = Callable[[ymfm.Chip, list[int]], None]

# Type alias for chip factory (any callable that takes clock and returns Chip)
ChipFactory = Callable[..., ymfm.Chip]


# Output parameters
SAMPLE_RATE = 48000
DURATION_SECONDS = 2
NUM_SAMPLES = SAMPLE_RATE * DURATION_SECONDS

# Target frequencies
FREQUENCIES = [110, 220, 440, 880]


def write_wav(filename: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write samples to a WAV file."""
    # Normalize to 16-bit range with headroom
    max_val = np.max(np.abs(samples))
    if max_val > 0:
        samples_16 = (samples * 30000 // max_val).astype(np.int16)
    else:
        samples_16 = samples.astype(np.int16)

    # Ensure stereo
    if len(samples_16.shape) == 1:
        samples_16 = np.column_stack([samples_16, samples_16])

    with wave.open(filename, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_16.tobytes())


def resample_to_48k(samples: np.ndarray, native_rate: int) -> np.ndarray:
    """Resample from native chip rate to 48kHz using linear interpolation."""
    if native_rate == SAMPLE_RATE:
        return samples

    # Calculate number of output samples
    native_samples = len(samples)
    output_samples = int(native_samples * SAMPLE_RATE / native_rate)

    # Create output array
    if len(samples.shape) == 1:
        output = np.zeros(output_samples, dtype=np.int32)
        indices = np.linspace(0, native_samples - 1, output_samples)
        output = np.interp(indices, np.arange(native_samples), samples).astype(np.int32)
    else:
        num_channels = samples.shape[1]
        output = np.zeros((output_samples, num_channels), dtype=np.int32)
        for ch in range(num_channels):
            indices = np.linspace(0, native_samples - 1, output_samples)
            output[:, ch] = np.interp(
                indices, np.arange(native_samples), samples[:, ch]
            ).astype(np.int32)

    return output


# =============================================================================
# OPL Family (YM3812, YMF262, YM3526, Y8950, YM2413)
# =============================================================================


def calc_opl_fnumber_block(freq: float, sample_rate: int) -> tuple[int, int]:
    """
    Calculate OPL F-number and block for a given frequency.

    OPL frequency formula: F = Fnum * Fsample / 2^(20-Block)
    Rearranged: Fnum = F * 2^(20-Block) / Fsample

    Returns (fnum, block) where fnum is 10-bit (0-1023) and block is 3-bit (0-7).
    """
    # Find the best block that keeps fnum in valid range (1-1023)
    for block in range(8):
        fnum = int(freq * (1 << (20 - block)) / sample_rate)
        if 1 <= fnum <= 0x3FF:
            return fnum, block
    # Fallback: use highest block, clamp fnum
    fnum = int(freq * (1 << (20 - 7)) / sample_rate)
    return min(fnum, 0x3FF), 7


def calc_opl_fnumber(freq: float, sample_rate: int, block: int = 4) -> int:
    """Legacy wrapper - prefer calc_opl_fnumber_block for correct results."""
    fnum = int(freq * (1 << (20 - block)) / sample_rate)
    return min(fnum, 0x3FF)  # 10-bit F-number


def setup_ym3812(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM3812 (OPL2) to generate sine waves.

    OPL2 has 9 channels, each with 2 operators.
    We use algorithm 0 (op1 modulates op2) with op1 at max attenuation
    to get a pure sine wave from op2.
    """
    sample_rate = chip.sample_rate
    chip.reset()

    # Enable waveform selection (required for OPL2)
    chip.write(0, 0x01)  # Test register address
    chip.write(1, 0x20)  # Enable waveform select

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        # Operator offsets for this channel
        # OPL2 operator mapping: ch 0-2 -> op 0-2,3-5; ch 3-5 -> op 6-8,9-11; etc.
        op1_offset = (ch % 3) + (ch // 3) * 8
        op2_offset = op1_offset + 3

        # Calculate F-number and block automatically
        fnum, block = calc_opl_fnumber_block(freq, sample_rate)

        # Operator 1 (modulator) - set to max attenuation (silent)
        chip.write(0, 0x20 + op1_offset)  # AM/VIB/EGT/KSR/MULT
        chip.write(1, 0x01)  # MULT=1, no tremolo/vibrato
        chip.write(0, 0x40 + op1_offset)  # KSL/TL (total level)
        chip.write(1, 0x3F)  # Max attenuation (silent)
        chip.write(0, 0x60 + op1_offset)  # AR/DR
        chip.write(1, 0xF0)  # Fast attack, no decay
        chip.write(0, 0x80 + op1_offset)  # SL/RR
        chip.write(1, 0x00)  # Sustain=0, no release
        chip.write(0, 0xE0 + op1_offset)  # Waveform
        chip.write(1, 0x00)  # Sine wave

        # Operator 2 (carrier) - audible sine wave
        chip.write(0, 0x20 + op2_offset)
        chip.write(1, 0x01)  # MULT=1
        chip.write(0, 0x40 + op2_offset)
        chip.write(1, 0x00)  # No attenuation (full volume)
        chip.write(0, 0x60 + op2_offset)
        chip.write(1, 0xF0)  # Fast attack
        chip.write(0, 0x80 + op2_offset)
        chip.write(1, 0x00)  # Sustain
        chip.write(0, 0xE0 + op2_offset)
        chip.write(1, 0x00)  # Sine wave

        # Channel frequency
        chip.write(0, 0xA0 + ch)  # F-number low
        chip.write(1, fnum & 0xFF)
        chip.write(0, 0xB0 + ch)  # Key on + block + F-number high
        chip.write(1, 0x20 | (block << 2) | ((fnum >> 8) & 0x03))

        # Feedback/connection
        chip.write(0, 0xC0 + ch)
        chip.write(1, 0x00)  # No feedback, algorithm 0


def setup_ymf262(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YMF262 (OPL3) to generate sine waves.

    Similar to OPL2 but with 18 channels and 4 outputs.
    """
    sample_rate = chip.sample_rate
    chip.reset()

    # Enable OPL3 mode
    chip.write(2, 0x05)  # Register 0x105
    chip.write(3, 0x01)  # NEW=1 (enable OPL3)

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        op1_offset = (ch % 3) + (ch // 3) * 8
        op2_offset = op1_offset + 3

        # Calculate F-number and block automatically
        fnum, block = calc_opl_fnumber_block(freq, sample_rate)

        # Operator 1 (modulator) - silent
        chip.write(0, 0x20 + op1_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op1_offset)
        chip.write(1, 0x3F)
        chip.write(0, 0x60 + op1_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op1_offset)
        chip.write(1, 0x00)
        chip.write(0, 0xE0 + op1_offset)
        chip.write(1, 0x00)

        # Operator 2 (carrier) - audible
        chip.write(0, 0x20 + op2_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0x60 + op2_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0xE0 + op2_offset)
        chip.write(1, 0x00)

        # Frequency
        chip.write(0, 0xA0 + ch)
        chip.write(1, fnum & 0xFF)
        chip.write(0, 0xB0 + ch)
        chip.write(1, 0x20 | (block << 2) | ((fnum >> 8) & 0x03))

        # Output to both left and right
        chip.write(0, 0xC0 + ch)
        chip.write(1, 0x30)  # Left + Right output


def setup_ym3526(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """Set up YM3526 (OPL) - same as OPL2 but no waveform selection."""
    sample_rate = chip.sample_rate
    chip.reset()

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        op1_offset = (ch % 3) + (ch // 3) * 8
        op2_offset = op1_offset + 3

        # Calculate F-number and block automatically
        fnum, block = calc_opl_fnumber_block(freq, sample_rate)

        # Operator 1 - silent
        chip.write(0, 0x20 + op1_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op1_offset)
        chip.write(1, 0x3F)
        chip.write(0, 0x60 + op1_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op1_offset)
        chip.write(1, 0x00)

        # Operator 2 - audible
        chip.write(0, 0x20 + op2_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0x60 + op2_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op2_offset)
        chip.write(1, 0x00)

        # Frequency and key on
        chip.write(0, 0xA0 + ch)
        chip.write(1, fnum & 0xFF)
        chip.write(0, 0xB0 + ch)
        chip.write(1, 0x20 | (block << 2) | ((fnum >> 8) & 0x03))

        chip.write(0, 0xC0 + ch)
        chip.write(1, 0x00)


def setup_y8950(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """Set up Y8950 (MSX-Audio) - same as OPL."""
    setup_ym3526(chip, frequencies)


def calc_opll_fnumber_block(freq: float, clock: int) -> tuple[int, int]:
    """
    Calculate OPLL (YM2413) F-number and block for a given frequency.

    OPLL frequency formula: F = Fnum * Clock / (72 * 2^(19-Block))
    Rearranged: Fnum = F * 72 * 2^(19-Block) / Clock

    Returns (fnum, block) where fnum is 9-bit (0-511) and block is 3-bit (0-7).
    """
    # Find the best block that keeps fnum in valid range (1-511)
    for block in range(8):
        fnum = int(freq * 72 * (1 << (19 - block)) / clock)
        if 1 <= fnum <= 0x1FF:
            return fnum, block
    # Fallback: use highest block, clamp fnum
    fnum = int(freq * 72 * (1 << (19 - 7)) / clock)
    return min(fnum, 0x1FF), 7


def calc_opll_fnumber(freq: float, clock: int, block: int = 4) -> int:
    """Legacy wrapper - prefer calc_opll_fnumber_block for correct results."""
    fnum = int(freq * 72 * (1 << (19 - block)) / clock)
    return min(fnum, 0x1FF)  # 9-bit F-number


def setup_ym2413(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2413 (OPLL) to generate sine waves.

    OPLL has built-in instruments. Instrument 0 is user-defined.
    We program it to output a simple sine.
    """
    clock = chip.clock
    chip.reset()

    # Program custom instrument (registers 0x00-0x07)
    # Modulator settings - minimize FM modulation for cleaner sine
    chip.write(0, 0x00)
    chip.write(1, 0x20)  # EGT=1 (sustain), MULT=0 (0.5x freq, less audible)
    chip.write(0, 0x02)
    chip.write(1, 0x3F)  # KSL=0, TL=63 (max attenuation)
    chip.write(0, 0x04)
    chip.write(1, 0xFF)  # AR=15 (instant), DR=15 (fast decay)
    chip.write(0, 0x06)
    chip.write(1, 0xFF)  # SL=15 (min sustain), RR=15 (fast release)

    # Carrier settings (audible sine wave)
    chip.write(0, 0x01)
    chip.write(1, 0x21)  # EGT=1 (sustain), MULT=1
    chip.write(0, 0x03)
    chip.write(1, 0x00)  # KSL=0, DC=0, DM=0, FB=0 (no feedback)
    chip.write(0, 0x05)
    chip.write(1, 0xF0)  # AR=15 (instant), DR=0 (no decay)
    chip.write(0, 0x07)
    chip.write(1, 0x0F)  # SL=0 (max sustain), RR=15

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        # Calculate F-number and block automatically
        fnum, block = calc_opll_fnumber_block(freq, clock)

        # Frequency low (8 bits)
        chip.write(0, 0x10 + ch)
        chip.write(1, fnum & 0xFF)

        # Key on + sustain + block + F-number high (bit 8)
        chip.write(0, 0x20 + ch)
        chip.write(1, 0x30 | (block << 1) | ((fnum >> 8) & 0x01))

        # Instrument 0 (custom) + volume
        chip.write(0, 0x30 + ch)
        chip.write(1, 0x00)  # Instrument 0, volume 0 (max)


# =============================================================================
# OPN Family (YM2612, YM2203, YM2608, YM2610)
# =============================================================================


def calc_opn_fnumber_block(freq: float, clock: int, prescaler: int) -> tuple[int, int]:
    """
    Calculate OPN F-number and block for a given frequency.

    OPN frequency formula: F = Fnum * Clock / (Prescaler * 2^(21-Block))
    Rearranged: Fnum = F * Prescaler * 2^(21-Block) / Clock

    Returns (fnum, block) where fnum is 11-bit (0-2047) and block is 3-bit (0-7).

    Prescaler values:
    - YM2203: 72 (6 * 12)
    - YM2612/YM3438/YM2608/YM2610/YM2610B: 144 (6 * 24)
    """
    # Find the best block that keeps fnum in valid range (1-2047)
    for block in range(8):
        fnum = int(freq * prescaler * (1 << (21 - block)) / clock)
        if 1 <= fnum <= 0x7FF:
            return fnum, block
    # Fallback: use highest block, clamp fnum
    fnum = int(freq * prescaler * (1 << (21 - 7)) / clock)
    return min(fnum, 0x7FF), 7


def calc_opn_fnumber(freq: float, clock: int, prescaler: int, block: int = 4) -> int:
    """Legacy wrapper - prefer calc_opn_fnumber_block for correct results."""
    fnum = int(freq * prescaler * (1 << (21 - block)) / clock)
    return min(fnum, 0x7FF)  # 11-bit F-number


def setup_ym2612(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2612 (OPN2) to generate sine waves.

    YM2612 has 6 FM channels, each with 4 operators.
    Use algorithm 7 (all operators output directly) with only OP1 active.
    """
    clock = chip.clock
    chip.reset()

    # Disable LFO
    chip.write(0, 0x22)
    chip.write(1, 0x00)

    # Enable all channels
    chip.write(0, 0x2B)
    chip.write(1, 0x00)  # DAC disable

    for ch, freq in enumerate(frequencies[:6]):
        if ch >= 6:
            break

        # Determine port (0-2 on port 0, 3-5 on port 1)
        port = 0 if ch < 3 else 2
        ch_offset = ch % 3

        # Calculate F-number and block automatically
        fnum, block = calc_opn_fnumber_block(freq, clock, 144)  # YM2612 prescaler = 144

        # Set algorithm 7 (all carriers) and feedback 0
        chip.write(port, 0xB0 + ch_offset)
        chip.write(port + 1, 0x07)  # Algorithm 7, feedback 0

        # Set up only operator 1 (slot 0) as audible
        for op in range(4):
            op_offset = ch_offset + [0, 8, 4, 12][op]

            # Detune/Multiple
            chip.write(port, 0x30 + op_offset)
            chip.write(port + 1, 0x01 if op == 0 else 0x00)  # MULT=1 for OP1

            # Total Level
            chip.write(port, 0x40 + op_offset)
            chip.write(
                port + 1, 0x00 if op == 0 else 0x7F
            )  # OP1 audible, others silent

            # Rate Scaling / Attack Rate
            chip.write(port, 0x50 + op_offset)
            chip.write(port + 1, 0x1F)  # Fast attack

            # AM / Decay Rate
            chip.write(port, 0x60 + op_offset)
            chip.write(port + 1, 0x00)  # No decay

            # Sustain Rate
            chip.write(port, 0x70 + op_offset)
            chip.write(port + 1, 0x00)  # No sustain decay

            # Sustain Level / Release Rate
            chip.write(port, 0x80 + op_offset)
            chip.write(port + 1, 0x00)  # Sustain level 0, slow release

        # Frequency
        chip.write(port, 0xA4 + ch_offset)  # Block/F-num high
        chip.write(port + 1, (block << 3) | ((fnum >> 8) & 0x07))
        chip.write(port, 0xA0 + ch_offset)  # F-num low
        chip.write(port + 1, fnum & 0xFF)

        # Stereo output
        chip.write(port, 0xB4 + ch_offset)
        chip.write(port + 1, 0xC0)  # Left + Right

        # Key on (all operators)
        chip.write(0, 0x28)
        chip.write(1, 0xF0 | (ch if ch < 3 else ch + 1))


def setup_ym3438(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YM3438 is register-compatible with YM2612."""
    setup_ym2612(chip, frequencies)


def setup_ym2203(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2203 (OPN) to generate sine waves.

    YM2203 has 3 FM channels + SSG. With algorithm 7 (all operators as carriers),
    we can use different MULT values to generate 4 frequencies from one channel.
    Base frequency 110Hz with MULT 1,2,4,8 gives 110,220,440,880 Hz.
    """
    clock = chip.clock
    chip.reset()

    # Use channel 0 with all 4 operators as carriers (algorithm 7)
    # Each operator uses a different MULT to generate different frequencies
    ch = 0

    # Base frequency is the lowest (110Hz), others are multiples
    base_freq = frequencies[0] if frequencies else 110

    # Calculate F-number and block for base frequency
    fnum, block = calc_opn_fnumber_block(base_freq, clock, 72)  # YM2203 prescaler = 72

    # Algorithm 7 (all carriers), feedback 0
    chip.write(0, 0xB0 + ch)
    chip.write(1, 0x07)

    # MULT values to generate 110, 220, 440, 880 Hz from 110Hz base
    # MULT: 0=0.5x, 1=1x, 2=2x, 3=3x, 4=4x, 5=5x, 6=6x, 7=7x, 8=8x, ...
    mult_values = [1, 2, 4, 8]  # For 110, 220, 440, 880 Hz

    # Set up all 4 operators as carriers with different MULT
    for op in range(4):
        op_offset = ch + [0, 8, 4, 12][op]

        # DT1/MUL - set multiplier for this operator
        mult = mult_values[op] if op < len(mult_values) else 1
        chip.write(0, 0x30 + op_offset)
        chip.write(1, mult & 0x0F)  # DT1=0, MUL=mult

        # TL (Total Level) - all operators audible
        chip.write(0, 0x40 + op_offset)
        chip.write(1, 0x00)  # Full volume

        # RS/AR
        chip.write(0, 0x50 + op_offset)
        chip.write(1, 0x1F)  # Fast attack

        # AM/D1R
        chip.write(0, 0x60 + op_offset)
        chip.write(1, 0x00)

        # D2R
        chip.write(0, 0x70 + op_offset)
        chip.write(1, 0x00)

        # D1L/RR
        chip.write(0, 0x80 + op_offset)
        chip.write(1, 0x00)

    # Frequency (shared by all operators, MULT determines actual freq)
    chip.write(0, 0xA4 + ch)
    chip.write(1, (block << 3) | ((fnum >> 8) & 0x07))
    chip.write(0, 0xA0 + ch)
    chip.write(1, fnum & 0xFF)

    # Key on (all operators)
    chip.write(0, 0x28)
    chip.write(1, 0xF0 | ch)


def setup_ym2608(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2608 (OPNA) to generate sine waves.

    YM2608 has 6 FM channels + SSG + ADPCM.
    Register 0x29 bit 7 must be set to enable the second FM bank (channels 3-5).
    """
    clock = chip.clock
    chip.reset()

    # Enable 6-channel mode (bit 7 of register 0x29 enables second FM bank)
    chip.write(0, 0x29)
    chip.write(1, 0x80)

    for ch, freq in enumerate(frequencies[:6]):
        if ch >= 6:
            break

        port = 0 if ch < 3 else 2
        ch_offset = ch % 3

        # Calculate F-number and block automatically
        fnum, block = calc_opn_fnumber_block(freq, clock, 144)  # YM2608 prescaler = 144

        # Algorithm 7 (all carriers), feedback 0
        chip.write(port, 0xB0 + ch_offset)
        chip.write(port + 1, 0x07)

        # Set up only operator 1 as carrier
        for op in range(4):
            op_offset = ch_offset + [0, 8, 4, 12][op]

            chip.write(port, 0x30 + op_offset)
            chip.write(port + 1, 0x01 if op == 0 else 0x00)

            chip.write(port, 0x40 + op_offset)
            chip.write(port + 1, 0x00 if op == 0 else 0x7F)

            chip.write(port, 0x50 + op_offset)
            chip.write(port + 1, 0x1F)

            chip.write(port, 0x60 + op_offset)
            chip.write(port + 1, 0x00)

            chip.write(port, 0x70 + op_offset)
            chip.write(port + 1, 0x00)

            chip.write(port, 0x80 + op_offset)
            chip.write(port + 1, 0x00)

        # Frequency
        chip.write(port, 0xA4 + ch_offset)
        chip.write(port + 1, (block << 3) | ((fnum >> 8) & 0x07))
        chip.write(port, 0xA0 + ch_offset)
        chip.write(port + 1, fnum & 0xFF)

        # Stereo output
        chip.write(port, 0xB4 + ch_offset)
        chip.write(port + 1, 0xC0)

        # Key on
        chip.write(0, 0x28)
        chip.write(1, 0xF0 | (ch if ch < 3 else ch + 1))


def setup_ym2610(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2610 (OPNB) to generate sine waves.

    YM2610 has 4 FM channels. Channel 0 is reserved on both banks,
    so we use channels 1, 2 on bank 0 and channels 1, 2 on bank 1.
    """
    clock = chip.clock
    chip.reset()

    # YM2610 FM channels: 1, 2 on port 0 (bank 0); 1, 2 on port 2 (bank 1)
    # Channel 0 is reserved on both banks for ADPCM
    channel_map = [(0, 1), (0, 2), (2, 1), (2, 2)]

    for idx, freq in enumerate(frequencies[:4]):
        if idx >= 4:
            break

        port, ch_offset = channel_map[idx]

        # Calculate F-number and block automatically
        fnum, block = calc_opn_fnumber_block(freq, clock, 144)  # YM2610 prescaler = 144

        chip.write(port, 0xB0 + ch_offset)
        chip.write(port + 1, 0x07)

        for op in range(4):
            op_offset = ch_offset + [0, 8, 4, 12][op]

            chip.write(port, 0x30 + op_offset)
            chip.write(port + 1, 0x01 if op == 0 else 0x00)

            chip.write(port, 0x40 + op_offset)
            chip.write(port + 1, 0x00 if op == 0 else 0x7F)

            chip.write(port, 0x50 + op_offset)
            chip.write(port + 1, 0x1F)

            chip.write(port, 0x60 + op_offset)
            chip.write(port + 1, 0x00)

            chip.write(port, 0x70 + op_offset)
            chip.write(port + 1, 0x00)

            chip.write(port, 0x80 + op_offset)
            chip.write(port + 1, 0x00)

        chip.write(port, 0xA4 + ch_offset)
        chip.write(port + 1, (block << 3) | ((fnum >> 8) & 0x07))
        chip.write(port, 0xA0 + ch_offset)
        chip.write(port + 1, fnum & 0xFF)

        chip.write(port, 0xB4 + ch_offset)
        chip.write(port + 1, 0xC0)

        # Key on: bank 0 channels are 1,2; bank 1 channels are 5,6
        key_ch = [1, 2, 5, 6][idx]
        chip.write(0, 0x28)
        chip.write(1, 0xF0 | key_ch)


def setup_ym2610b(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YM2610B has 6 FM channels like YM2608."""
    setup_ym2608(chip, frequencies)


# =============================================================================
# OPM Family (YM2151, YM2164)
# =============================================================================


def calc_opm_kc_kf(freq: float, clock: int) -> tuple[int, int]:
    """
    Calculate OPM KC (Key Code) and KF (Key Fraction) for a frequency.

    OPM KC encoding:
    - Bits 6-4: Octave (0-7)
    - Bits 3-0: Note (0-15, but only 0,1,2,4,5,6,8,9,10,12,13,14 are valid)

    The note values map to semitones with gaps at 3,7,11,15.
    """
    if freq <= 0:
        return 0, 0

    # Reference: C#4 (277.18Hz) is at KC=0x40 (octave 4, note 0) at 3.58MHz
    # A4 (440Hz) is at KC=0x4A (octave 4, note 10)
    ref_freq = 277.18  # C#4
    ref_octave = 4
    ref_note = 0

    # Calculate semitones from reference
    semitones_from_ref = 12 * np.log2(freq / ref_freq)

    # Split into octaves and semitones within octave
    total_semitones = ref_note + semitones_from_ref
    octave = ref_octave + int(total_semitones // 12)
    semitone_in_octave = total_semitones % 12

    # Handle negative semitones
    while semitone_in_octave < 0:
        semitone_in_octave += 12
        octave -= 1

    # Map semitone (0-11) to OPM note value (skipping 3,7,11,15)
    # Semitone:  0  1  2  3  4  5  6  7  8  9  10 11
    # OPM note:  0  1  2  4  5  6  8  9  10 12 13 14
    note_map = [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]
    note_idx = int(semitone_in_octave)
    note = note_map[note_idx % 12]

    # Calculate KF (fine tuning)
    frac = semitone_in_octave - note_idx
    kf = int(frac * 64) & 0x3F

    # Build KC
    octave = max(0, min(7, octave))
    kc = (octave << 4) | note

    return kc, kf


def setup_ym2151(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2151 (OPM) to generate sine waves.

    YM2151 has 8 channels, each with 4 operators.
    """
    clock = chip.clock
    chip.reset()

    # Disable LFO
    chip.write(0, 0x18)  # LFRQ
    chip.write(1, 0x00)
    chip.write(0, 0x19)  # PMD/AMD
    chip.write(1, 0x00)

    for ch, freq in enumerate(frequencies[:8]):
        if ch >= 8:
            break

        kc, kf = calc_opm_kc_kf(freq, clock)

        # Connection (algorithm) and feedback
        chip.write(0, 0x20 + ch)  # RL/FB/CONNECT
        chip.write(1, 0xC7)  # Both outputs, no feedback, algorithm 7

        # Set up operators
        for op in range(4):
            op_offset = ch + op * 8

            # DT1/MUL
            chip.write(0, 0x40 + op_offset)
            chip.write(1, 0x01 if op == 0 else 0x00)

            # TL (Total Level)
            chip.write(0, 0x60 + op_offset)
            chip.write(1, 0x00 if op == 0 else 0x7F)

            # KS/AR (Key Scale/Attack Rate)
            chip.write(0, 0x80 + op_offset)
            chip.write(1, 0x1F)  # Fast attack

            # AMS-EN/D1R
            chip.write(0, 0xA0 + op_offset)
            chip.write(1, 0x00)

            # DT2/D2R
            chip.write(0, 0xC0 + op_offset)
            chip.write(1, 0x00)

            # D1L/RR
            chip.write(0, 0xE0 + op_offset)
            chip.write(1, 0x00)

        # Key Code
        chip.write(0, 0x28 + ch)
        chip.write(1, kc)

        # Key Fraction
        chip.write(0, 0x30 + ch)
        chip.write(1, kf << 2)

        # Key On (all operators)
        chip.write(0, 0x08)
        chip.write(1, 0x78 | ch)  # SN=0111 (all ops), CH=ch


def setup_ym2164(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YM2164 is register-compatible with YM2151."""
    setup_ym2151(chip, frequencies)


# =============================================================================
# Additional OPL/OPN chips (Phase 3)
# =============================================================================


def setup_ymf262_variant(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """Set up OPL3 variants (YMF289B) - same as YMF262."""
    sample_rate = chip.sample_rate
    chip.reset()

    # Enable OPL3 mode
    chip.write(2, 0x05)
    chip.write(3, 0x01)

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        op1_offset = (ch % 3) + (ch // 3) * 8
        op2_offset = op1_offset + 3

        fnum, block = calc_opl_fnumber_block(freq, sample_rate)

        chip.write(0, 0x20 + op1_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op1_offset)
        chip.write(1, 0x3F)
        chip.write(0, 0x60 + op1_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op1_offset)
        chip.write(1, 0x00)
        chip.write(0, 0xE0 + op1_offset)
        chip.write(1, 0x00)

        chip.write(0, 0x20 + op2_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0x60 + op2_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0xE0 + op2_offset)
        chip.write(1, 0x00)

        chip.write(0, 0xA0 + ch)
        chip.write(1, fnum & 0xFF)
        chip.write(0, 0xB0 + ch)
        chip.write(1, 0x20 | (block << 2) | ((fnum >> 8) & 0x03))

        chip.write(0, 0xC0 + ch)
        chip.write(1, 0x30)


def setup_ymf289b(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YMF289B (OPL3L) is OPL3 compatible."""
    setup_ymf262_variant(chip, frequencies)


def setup_ymf278b(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YMF278B (OPL4) FM part is OPL3 compatible.

    Note: YMF278B.sample_rate returns the PCM rate (44100Hz), not the FM rate.
    The FM sample rate is clock / 684 (prescaler=19, divider=36), not clock / 288.
    """
    # Calculate FM sample rate for OPL4: clock / 684
    # (OPL3 uses clock / 288, but OPL4 has different prescaler)
    fm_sample_rate = chip.clock // 684
    chip.reset()

    # Enable OPL3 mode
    chip.write(2, 0x05)
    chip.write(3, 0x01)

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        op1_offset = (ch % 3) + (ch // 3) * 8
        op2_offset = op1_offset + 3

        fnum, block = calc_opl_fnumber_block(freq, fm_sample_rate)

        chip.write(0, 0x20 + op1_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op1_offset)
        chip.write(1, 0x3F)
        chip.write(0, 0x60 + op1_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op1_offset)
        chip.write(1, 0x00)
        chip.write(0, 0xE0 + op1_offset)
        chip.write(1, 0x00)

        chip.write(0, 0x20 + op2_offset)
        chip.write(1, 0x01)
        chip.write(0, 0x40 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0x60 + op2_offset)
        chip.write(1, 0xF0)
        chip.write(0, 0x80 + op2_offset)
        chip.write(1, 0x00)
        chip.write(0, 0xE0 + op2_offset)
        chip.write(1, 0x00)

        chip.write(0, 0xA0 + ch)
        chip.write(1, fnum & 0xFF)
        chip.write(0, 0xB0 + ch)
        chip.write(1, 0x20 | (block << 2) | ((fnum >> 8) & 0x03))

        chip.write(0, 0xC0 + ch)
        chip.write(1, 0x30)


def setup_opll_variant(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """Set up OPLL variants (YM2423, YMF281, DS1001) - same as YM2413."""
    clock = chip.clock
    chip.reset()

    # Program custom instrument
    chip.write(0, 0x00)
    chip.write(1, 0x20)
    chip.write(0, 0x02)
    chip.write(1, 0x3F)
    chip.write(0, 0x04)
    chip.write(1, 0xFF)
    chip.write(0, 0x06)
    chip.write(1, 0xFF)

    chip.write(0, 0x01)
    chip.write(1, 0x21)
    chip.write(0, 0x03)
    chip.write(1, 0x00)
    chip.write(0, 0x05)
    chip.write(1, 0xF0)
    chip.write(0, 0x07)
    chip.write(1, 0x0F)

    for ch, freq in enumerate(frequencies[:9]):
        if ch >= 9:
            break

        fnum, block = calc_opll_fnumber_block(freq, clock)

        chip.write(0, 0x10 + ch)
        chip.write(1, fnum & 0xFF)
        chip.write(0, 0x20 + ch)
        chip.write(1, 0x30 | (block << 1) | ((fnum >> 8) & 0x01))
        chip.write(0, 0x30 + ch)
        chip.write(1, 0x00)


def setup_ym2423(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YM2423 is OPLL compatible."""
    setup_opll_variant(chip, frequencies)


def setup_ymf281(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YMF281 is OPLL compatible."""
    setup_opll_variant(chip, frequencies)


def setup_ds1001(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """DS1001 (VRC7) is OPLL compatible."""
    setup_opll_variant(chip, frequencies)


def setup_ymf276(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YMF276 is YM2612 compatible."""
    setup_ym2612(chip, frequencies)


def setup_ymf288(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YMF288 (OPN3L) is similar to YM2608."""
    setup_ym2608(chip, frequencies)


# =============================================================================
# Misc chip families (SSG, OPZ, OPQ)
# =============================================================================


def setup_ym2149(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2149 (SSG/PSG) to generate square waves.

    SSG generates square waves, not sine waves.
    3 channels available (A, B, C).

    Note: YM2149 uses write_address/write_data protocol, not write(offset, data).

    SSG timing: sample_rate = clock / 64, tone toggles every TP samples.
    So frequency = sample_rate / (2 * TP) = clock / (128 * TP)
    Therefore: TP = clock / (128 * freq)
    """
    clock = chip.clock
    chip.reset()

    for ch, freq in enumerate(frequencies[:3]):
        if ch >= 3:
            break

        # SSG frequency = Clock / (128 * TP)
        # TP = Clock / (128 * freq)
        tp = int(clock / (128 * freq))
        tp = max(1, min(0xFFF, tp))  # 12-bit value

        # Fine tune register (low 8 bits)
        chip.write_address(0x00 + ch * 2)
        chip.write_data(tp & 0xFF)
        # Coarse tune register (high 4 bits)
        chip.write_address(0x01 + ch * 2)
        chip.write_data((tp >> 8) & 0x0F)

        # Set volume (max)
        chip.write_address(0x08 + ch)
        chip.write_data(0x0F)

    # Enable tone on channels A, B, C (disable noise)
    chip.write_address(0x07)  # Mixer register
    chip.write_data(0x38)  # Enable tone A,B,C; disable noise


def setup_ym2414(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM2414 (OPZ) to generate sine waves.

    OPZ register layout (from ymfm_opz.h):
    - Register 0x08: Channel selection (bits 0-2) - MUST be set before key-on!
    - Channel volume: 0x00-0x07 (8-bit per-channel volume)
    - Register 0x20+ch: output/key-on/FB/ALG
    - Output enable: bit 7 of 0x20+ch for output 1, bit 0 of 0x30+ch for output 0
    - Register 0x30+ch: KF (key fraction) + output 0 enable
    - Per-operator: channel in bits 0-2, operator in bits 3-4

    Register 0x20+ch layout:
      bit 7: output 1 enable (right)
      bit 6: key on (1=ON, 0=OFF)
      bits 5-3: feedback
      bits 2-0: algorithm

    CRITICAL: Key-on only works when register 0x08 contains the target channel!
    The ymfm implementation requires: bitfield(0x08, 0, 3) == channel number
    before writing to 0x20+ch with key-on bit set.
    """
    clock = chip.clock
    chip.reset()

    # Disable LFO
    chip.write_address(0x18)  # LFO rate
    chip.write_data(0x00)
    chip.write_address(0x19)  # LFO depth (AMD)
    chip.write_data(0x00)

    for ch, freq in enumerate(frequencies[:8]):
        if ch >= 8:
            break

        kc, kf = calc_opm_kc_kf(freq, clock)

        # Set channel volume to max (registers 0x00-0x07)
        chip.write_address(0x00 + ch)
        chip.write_data(0xFF)  # Max volume

        # Set up operators first (before key on)
        for op in range(4):
            op_offset = ch + op * 8

            chip.write_address(0x40 + op_offset)  # DT1/MUL
            chip.write_data(0x01 if op == 0 else 0x00)

            chip.write_address(0x60 + op_offset)  # TL
            chip.write_data(0x00 if op == 0 else 0x7F)

            chip.write_address(0x80 + op_offset)  # KS/AR
            chip.write_data(0x1F)

            chip.write_address(0xA0 + op_offset)  # AMS/D1R
            chip.write_data(0x00)

            chip.write_address(0xC0 + op_offset)  # DT2/D2R
            chip.write_data(0x00)

            chip.write_address(0xE0 + op_offset)  # D1L/RR
            chip.write_data(0x00)

        # KC (pitch)
        chip.write_address(0x28 + ch)
        chip.write_data(kc)

        # KF + output 0 enable (bit 0 enables stereo left output)
        chip.write_address(0x30 + ch)
        chip.write_data((kf << 2) | 0x01)

        # CRITICAL: Select channel in register 0x08 before key-on!
        # The ymfm OPZ implementation only processes key-on when the channel
        # being written matches the channel stored in register 0x08.
        chip.write_address(0x08)
        chip.write_data(ch)

        # Output 1 enable (bit 7), key ON (bit 6 = 1), FB=0, algorithm=7
        # 0xC7 = 11000111 = output 1 on, key ON, FB=0, ALG=7
        chip.write_address(0x20 + ch)
        chip.write_data(0xC7)


def calc_opq_fnumber_block(freq: float, clock: int) -> tuple[int, int]:
    """
    Calculate OPQ F-number and block for a given frequency.

    OPQ formula (from ymfm_opq.cpp):
      phase_step = (fnum << block) >> 2
      F = phase_step * sample_rate / 2^20
        = fnum * 2^block * sample_rate / 2^22

    Rearranged: fnum = F * 2^(22-block) / sample_rate
                     = F * 64 * 2^(22-block) / clock

    Returns (fnum, block) where fnum is 12-bit and block is 3-bit.
    """
    prescaler = 64  # OPQ: sample_rate = clock / 64
    for block in range(8):
        fnum = int(freq * prescaler * (1 << (22 - block)) / clock)
        if 1 <= fnum <= 0xFFF:  # 12-bit FNUM
            return fnum, block
    fnum = int(freq * prescaler * (1 << (22 - 7)) / clock)
    return min(fnum, 0xFFF), 7


def setup_ym3806(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """
    Set up YM3806 (OPQ) to generate sine waves.

    OPQ register layout (from ymfm_opq.h):
    - Key on: 0x05 (bits 6-3: operators, bits 2-0: channel)
    - Per-channel: 0x10-0x17 (pan/feedback/algorithm)
    - Frequency: 0x20-0x27 (block/fnum high), 0x30-0x37 (fnum low)
    - Per-operator: 0x40+ (DT/MUL), 0x60+ (TL), 0x80+ (KSR/AR), etc.

    Operators are indexed: channel in bits 0-2, operator in bits 3-4

    Note: OPQ uses direct register writes, not address/data protocol!
    Use chip.write(addr, data) not chip.write(0, addr); chip.write(1, data)
    """
    clock = chip.clock
    chip.reset()

    # Enable LFO (bit 3 = 0 means enabled)
    chip.write(0x04, 0x00)

    for ch, freq in enumerate(frequencies[:8]):
        if ch >= 8:
            break

        fnum, block = calc_opq_fnumber_block(freq, clock)

        # Set algorithm 7 (all carriers), both outputs enabled
        # Register 0x10+ch: bits 7-6 = pan (11=both), bits 5-3 = FB, bits 2-0 = ALG
        chip.write(0x10 + ch, 0xC7)  # Both outputs, FB=0, ALG=7

        # Set frequency for operators 2&4 (registers 0x20, 0x30)
        chip.write(0x20 + ch, (block << 4) | ((fnum >> 8) & 0x0F))
        chip.write(0x30 + ch, fnum & 0xFF)

        # Set frequency for operators 1&3 (registers 0x28, 0x38)
        chip.write(0x28 + ch, (block << 4) | ((fnum >> 8) & 0x0F))
        chip.write(0x38 + ch, fnum & 0xFF)

        # Set up all 4 operators
        for op in range(4):
            # Operator offset: channel + (operator * 8)
            op_offset = ch + (op * 8)

            # Multiple (write with bit 7 = 1 to select MUL register)
            chip.write(
                0x40 + op_offset, 0x81 if op == 0 else 0x80
            )  # MUL=1 for op0, MUL=0 for others

            # Total Level (0 = max volume, 127 = silent)
            chip.write(
                0x60 + op_offset, 0x00 if op == 0 else 0x7F
            )  # Op0 audible, others silent

            # KSR/AR (Attack Rate)
            chip.write(0x80 + op_offset, 0x1F)  # KSR=0, AR=31 (fast)

            # AM/Waveform/DR (Decay Rate)
            chip.write(0xA0 + op_offset, 0x00)  # AM=0, Wave=0, DR=0

            # SR (Sustain Rate)
            chip.write(0xC0 + op_offset, 0x00)  # SR=0

            # SL/RR (Sustain Level / Release Rate)
            chip.write(0xE0 + op_offset, 0x0F)  # SL=0, RR=15

        # Key on: register 0x05
        # Bits 6-3: operator enable (F = all 4 operators)
        # Bits 2-0: channel number
        chip.write(0x05, 0x78 | ch)  # All operators on for this channel


def setup_ym3533(chip: ymfm.Chip, frequencies: list[int]) -> None:
    """YM3533 is OPQ compatible."""
    setup_ym3806(chip, frequencies)


# =============================================================================
# Main generation functions
# =============================================================================

CHIP_CONFIGS: dict[str, tuple[ChipFactory, int, SetupFunc]] = {
    # OPL Family
    "YM3812": (ymfm.YM3812, 3579545, setup_ym3812),
    "YMF262": (ymfm.YMF262, 14318180, setup_ymf262),
    "YM3526": (ymfm.YM3526, 3579545, setup_ym3526),
    "Y8950": (ymfm.Y8950, 3579545, setup_y8950),
    "YM2413": (ymfm.YM2413, 3579545, setup_ym2413),
    "YMF289B": (ymfm.YMF289B, 14318180, setup_ymf289b),
    "YMF278B": (ymfm.YMF278B, 33868800, setup_ymf278b),
    "YM2423": (ymfm.YM2423, 3579545, setup_ym2423),
    "YMF281": (ymfm.YMF281, 3579545, setup_ymf281),
    "DS1001": (ymfm.DS1001, 3579545, setup_ds1001),
    # OPN Family
    "YM2612": (ymfm.YM2612, 7670453, setup_ym2612),
    "YM3438": (ymfm.YM3438, 7670453, setup_ym3438),
    "YM2203": (ymfm.YM2203, 4000000, setup_ym2203),
    "YM2608": (ymfm.YM2608, 8000000, setup_ym2608),
    "YM2610": (ymfm.YM2610, 8000000, setup_ym2610),
    "YM2610B": (ymfm.YM2610B, 8000000, setup_ym2610b),
    "YMF276": (ymfm.YMF276, 7670453, setup_ymf276),
    "YMF288": (ymfm.YMF288, 8000000, setup_ymf288),
    # OPM Family
    "YM2151": (ymfm.YM2151, 3579545, setup_ym2151),
    "YM2164": (ymfm.YM2164, 3579545, setup_ym2164),
    # Misc Families
    # YM2149 has read_data instead of read_status, so we cast it
    "YM2149": (cast(ChipFactory, ymfm.YM2149), 2000000, setup_ym2149),
    "YM2414": (ymfm.YM2414, 3579545, setup_ym2414),
    "YM3806": (ymfm.YM3806, 3579545, setup_ym3806),
    "YM3533": (ymfm.YM3533, 3579545, setup_ym3533),
}


def generate_chip_output(
    chip_name: str, frequencies: list[int] = FREQUENCIES
) -> tuple[np.ndarray, int]:
    """
    Generate audio output for a specific chip.

    Returns (samples, native_sample_rate).
    """
    chip_class, clock, setup_func = CHIP_CONFIGS[chip_name]

    # Create chip
    chip = chip_class(clock=clock)

    # Set up to generate frequencies
    setup_func(chip, frequencies)

    # Calculate native samples needed
    native_rate = chip.sample_rate
    native_samples = int(native_rate * DURATION_SECONDS)

    # Generate samples (returns memoryview, convert to numpy array)
    raw_samples = chip.generate(native_samples)
    samples = np.asarray(raw_samples, dtype=np.int32).reshape(-1, chip.outputs)

    # Convert to stereo if mono
    if chip.outputs == 1:
        samples = np.column_stack([samples[:, 0], samples[:, 0]])
    elif chip.outputs > 2:
        # Mix down to stereo
        # Some chips (like YMF278B/OPL4) have multiple output pairs:
        # - YMF278B: Ch0-1=wavetable, Ch2-3=external, Ch4-5=FM
        # Sum all channels pairwise for left and right
        left = np.zeros(len(samples), dtype=np.int64)
        right = np.zeros(len(samples), dtype=np.int64)
        for ch in range(0, chip.outputs, 2):
            left += samples[:, ch].astype(np.int64)
            if ch + 1 < chip.outputs:
                right += samples[:, ch + 1].astype(np.int64)
            else:
                right += samples[:, ch].astype(np.int64)
        # Clip to int32 range
        left = np.clip(left, -2147483648, 2147483647).astype(np.int32)
        right = np.clip(right, -2147483648, 2147483647).astype(np.int32)
        samples = np.column_stack([left, right])

    return samples, native_rate


def generate_all_chips(output_dir: str = "output"):
    """Generate sine wave outputs for all chips."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for chip_name in CHIP_CONFIGS:
        print(f"Generating {chip_name}...")

        samples, native_rate = generate_chip_output(chip_name)
        print(f"  Native rate: {native_rate}Hz, samples: {len(samples)}")

        # Resample to 48kHz
        samples_48k = resample_to_48k(samples, native_rate)
        print(f"  Resampled to {SAMPLE_RATE}Hz: {len(samples_48k)} samples")

        # Write WAV file
        wav_path = output_path / f"{chip_name.lower()}_sine.wav"
        write_wav(str(wav_path), samples_48k)
        print(f"  Written to {wav_path}")


if __name__ == "__main__":
    generate_all_chips()
    print("\nDone! WAV files written to 'output/' directory.")
