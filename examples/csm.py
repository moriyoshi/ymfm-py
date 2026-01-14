#!/usr/bin/env python3
"""
WAV to VGM converter for FM chip CSM voice synthesis.

This script converts a WAV file to a VGM file that plays back using
true CSM (Composite Sinusoidal Modeling) hardware mode of FM chips.
CSM mode automatically triggers key-on events at Timer A overflow rate,
creating the characteristic "grainy" speech synthesis effect.

Based on csm_voice_encode_synthesis_python by Yasunori Shimura
https://github.com/yas-sim/csm_voice_encode_synthesis_python
Licensed under the MIT License. See LICENSE file for details.

Supported chips:
  - YM2151 (OPM) - CSM triggers all 8 channels, uses 4 for formants
  - YM2203 (OPN) - CSM triggers channel 3 only with per-operator frequencies
  - YM2608 (OPNA) - CSM triggers channel 3 only with per-operator frequencies
  - YM3526 (OPL) - CSW triggers all 9 channels, uses 4 for formants
  - YM3812 (OPL2) - CSW triggers all 9 channels, uses 4 for formants
  - Y8950 (MSX-Audio) - CSW triggers all 9 channels, uses 4 for formants

CSM mode specifics:
  - YM2151: Register 0x14 bit 7 enables CSM, Timer A overflow keys all channels
  - YM2203/YM2608: Register 0x27 bits 6 enables CSM on channel 3
  - YM3526/YM3812/Y8950: Register 0x08 bit 7 enables CSW, Timer 1 overflow keys all channels

Usage:
    python csm.py input.wav [output.vgm] [--chip CHIP]
"""

import sys
import os
import struct
import argparse
from abc import ABC, abstractmethod
from typing import Annotated, Callable, Tuple

import wave

import numpy as np
from numpy.typing import NDArray


# =============================================================================
# VGM Constants
# =============================================================================
VGM_MAGIC = b"Vgm "
VGM_VERSION = 0x161  # Version 1.61
VGM_SAMPLE_RATE = 44100

# VGM Command opcodes
VGM_CMD_YM2151 = 0x54
VGM_CMD_YM2203 = 0x55
VGM_CMD_YM2608_P0 = 0x56  # Port 0
VGM_CMD_YM2608_P1 = 0x57  # Port 1
VGM_CMD_YM3526 = 0x5B
VGM_CMD_YM3812 = 0x5A
VGM_CMD_Y8950 = 0x5C
VGM_CMD_WAIT_NTSC = 0x62  # Wait 735 samples (1/60 sec)
VGM_CMD_WAIT_16BIT = 0x61  # Wait n samples (16-bit)
VGM_CMD_WAIT_SHORT = 0x70  # Wait 1-16 samples (0x70-0x7F)
VGM_CMD_END = 0x66


# =============================================================================
# VGM Writer
# =============================================================================
class VGMWriter:
    """Writes VGM file format."""

    total_samples: int
    _chip_clock: int
    _chip_header_offset: int

    def __init__(self) -> None:
        self.commands = bytearray()
        self.total_samples = 0
        self._chip_clock = 0
        self._chip_header_offset = 0

    def set_chip_info(self, clock: int, header_offset: int) -> None:
        """Set chip clock and header offset."""
        self._chip_clock = clock
        self._chip_header_offset = header_offset

    def write_command(self, opcode: int, reg: int, data: int) -> None:
        """Write a raw VGM command."""
        self.commands.append(opcode & 0xFF)
        self.commands.append(reg & 0xFF)
        self.commands.append(data & 0xFF)

    def write_wait(self, samples: int) -> None:
        """Write wait command(s) for the specified number of samples."""
        self.total_samples += samples

        while samples > 0:
            if samples >= 65535:
                self.commands.append(VGM_CMD_WAIT_16BIT)
                self.commands.append(0xFF)
                self.commands.append(0xFF)
                samples -= 65535
            elif samples >= 735:
                self.commands.append(VGM_CMD_WAIT_NTSC)
                samples -= 735
            elif samples >= 17:
                self.commands.append(VGM_CMD_WAIT_16BIT)
                self.commands.append(samples & 0xFF)
                self.commands.append((samples >> 8) & 0xFF)
                samples = 0
            elif samples >= 1:
                wait_val = min(samples, 16)
                self.commands.append(VGM_CMD_WAIT_SHORT + wait_val - 1)
                samples -= wait_val

    def write_end(self) -> None:
        """Write end-of-data marker."""
        self.commands.append(VGM_CMD_END)

    def build_header(self) -> bytes:
        """Build the VGM file header."""
        header = bytearray(0x100)  # 256-byte header for v1.61

        # Magic identifier
        header[0x00:0x04] = VGM_MAGIC

        # EOF offset (relative to 0x04)
        eof_offset = len(header) + len(self.commands) - 0x04
        struct.pack_into("<I", header, 0x04, eof_offset)

        # Version
        struct.pack_into("<I", header, 0x08, VGM_VERSION)

        # GD3 offset (0 = no GD3 tag)
        struct.pack_into("<I", header, 0x14, 0)

        # Total samples
        struct.pack_into("<I", header, 0x18, self.total_samples)

        # Loop offset (0 = no loop)
        struct.pack_into("<I", header, 0x1C, 0)

        # Loop samples (0 = no loop)
        struct.pack_into("<I", header, 0x20, 0)

        # Recording rate
        struct.pack_into("<I", header, 0x24, VGM_SAMPLE_RATE)

        # VGM data offset (relative to 0x34)
        struct.pack_into("<I", header, 0x34, 0x100 - 0x34)

        # Set chip clock at appropriate offset
        if self._chip_header_offset > 0:
            struct.pack_into("<I", header, self._chip_header_offset, self._chip_clock)

        return bytes(header)

    def save(self, filename: str) -> None:
        """Save the VGM file."""
        with open(filename, "wb") as f:
            f.write(self.build_header())
            f.write(self.commands)


# =============================================================================
# Abstract FM Chip Base Class
# =============================================================================
class FMChip(ABC):
    """Abstract base class for FM sound chips with CSM mode support."""

    def __init__(self, vgm: VGMWriter):
        self.vgm = vgm
        vgm.set_chip_info(self.clock, self.vgm_header_offset)

    @property
    @abstractmethod
    def name(self) -> str:
        """Chip name for display."""
        pass

    @property
    @abstractmethod
    def clock(self) -> int:
        """Chip clock frequency in Hz."""
        pass

    @property
    @abstractmethod
    def vgm_header_offset(self) -> int:
        """Offset in VGM header where clock is stored."""
        pass

    @property
    @abstractmethod
    def max_channels(self) -> int:
        """Maximum number of channels/operators available for CSM."""
        pass

    @abstractmethod
    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        """Write a value to a chip register."""
        pass

    @abstractmethod
    def init_channels(self, num_channels: int) -> None:
        """Initialize channels for CSM playback."""
        pass

    @abstractmethod
    def enable_csm_mode(self, period_samples: int) -> None:
        """Enable CSM mode with Timer A/Timer 1 period in samples.

        Args:
            period_samples: The period between CSM triggers in samples at VGM_SAMPLE_RATE.
        """
        pass

    @abstractmethod
    def disable_csm_mode(self) -> None:
        """Disable CSM mode."""
        pass

    @abstractmethod
    def convert_frequency(self, freq: float) -> Tuple:
        """Convert frequency to chip-specific format. Returns chip-specific tuple."""
        pass

    @abstractmethod
    def update_frame(self, channel: int, freq_data: Tuple, tl: int) -> None:
        """Update a channel with new frequency and amplitude for one frame."""
        pass

    @abstractmethod
    def key_on(self, num_channels: int) -> None:
        """Trigger key-on for all active channels (manual CSM trigger)."""
        pass

    @abstractmethod
    def key_off(self, num_channels: int) -> None:
        """Turn off all active channels."""
        pass

    def convert_amplitude(self, amp: float, max_amp: float) -> int:
        """
        Convert amplitude to Total Level (TL).
        TL is an attenuation value: 0 = loudest, 127 = silent.
        Can be overridden for chips with different TL ranges.
        """
        if max_amp <= 0 or amp <= 0:
            return 127

        norm_amp = amp / max_amp
        if norm_amp <= 0.001:
            return 127

        # Convert to dB attenuation (127 steps ~ 96 dB range)
        db = 20 * np.log10(norm_amp)
        tl = int(-db * 127 / 96)
        return max(0, min(127, tl))


# =============================================================================
# YM2151 (OPM) Implementation with CSM Mode
# =============================================================================
class YM2151Chip(FMChip):
    """YM2151 (OPM) FM sound chip with true CSM mode support.

    CSM mode on YM2151:
    - Controlled by bit 7 of register 0x14
    - Timer A overflow triggers key-on for ALL channels simultaneously
    - Timer A is 10-bit (registers 0x10 and 0x11)
    - Timer period: (1024 - TA) * 64 / clock
    """

    # Register definitions
    REG_KEY_ON = 0x08
    REG_TIMER_A_HI = 0x10  # Timer A high 8 bits
    REG_TIMER_A_LO = 0x11  # Timer A low 2 bits
    REG_TIMER_B = 0x12  # Timer B 8 bits
    REG_TIMER_CTRL = 0x14  # Timer control and CSM mode
    REG_LFRQ = 0x18
    REG_PMD_AMD = 0x19
    REG_RL_FB_CON = 0x20
    REG_KC = 0x28
    REG_KF = 0x30
    REG_DT1_MUL = 0x40
    REG_TL = 0x60
    REG_KS_AR = 0x80
    REG_AMS_D1R = 0xA0
    REG_DT2_D2R = 0xC0
    REG_D1L_RR = 0xE0

    # Timer control bits
    TIMER_CSM_ENABLE = 0x80  # Bit 7: CSM mode enable
    TIMER_B_IRQ_ENABLE = 0x08  # Bit 3: Timer B IRQ enable
    TIMER_A_IRQ_ENABLE = 0x04  # Bit 2: Timer A IRQ enable
    TIMER_B_LOAD = 0x02  # Bit 1: Load and start Timer B
    TIMER_A_LOAD = 0x01  # Bit 0: Load and start Timer A
    TIMER_B_RESET = 0x20  # Bit 5: Reset Timer B flag
    TIMER_A_RESET = 0x10  # Bit 4: Reset Timer A flag

    # Gappy note encoding map
    NOTE_MAP = np.array([0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14])

    @property
    def name(self) -> str:
        return "YM2151"

    @property
    def clock(self) -> int:
        return 3579545

    @property
    def vgm_header_offset(self) -> int:
        return 0x30

    @property
    def max_channels(self) -> int:
        return 8

    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        self.vgm.write_command(VGM_CMD_YM2151, reg, data)

    def _calculate_timer_a(self, period_samples: int) -> int:
        """Calculate Timer A value for a given period in samples at 44100 Hz.

        Timer A period formula: (1024 - TA) * 64 / clock
        Solving for TA: TA = 1024 - (period_seconds * clock / 64)
        """
        period_seconds = period_samples / VGM_SAMPLE_RATE
        timer_val = int(1024 - (period_seconds * self.clock / 64))
        return max(0, min(1023, timer_val))

    def _setup_timer_a(self, period_samples: int) -> None:
        """Set up Timer A for CSM mode."""
        timer_val = self._calculate_timer_a(period_samples)
        # Timer A high 8 bits
        self.write_reg(self.REG_TIMER_A_HI, (timer_val >> 2) & 0xFF)
        # Timer A low 2 bits
        self.write_reg(self.REG_TIMER_A_LO, timer_val & 0x03)

    def init_channels(self, num_channels: int) -> None:
        # Disable LFO
        self.write_reg(self.REG_LFRQ, 0x00)
        self.write_reg(self.REG_PMD_AMD, 0x00)

        for ch in range(num_channels):
            # RL/FB/CON: Both outputs, no feedback, algorithm 7 (all carriers)
            self.write_reg(self.REG_RL_FB_CON + ch, 0xC7)

            # Set up all 4 operators for this channel
            for op in range(4):
                op_off = ch + op * 8

                if op == 0:
                    # Operator 0 (M1) - the one we'll use for sound
                    self.write_reg(self.REG_DT1_MUL + op_off, 0x01)  # DT1=0, MUL=1
                    self.write_reg(self.REG_TL + op_off, 0x00)  # Start at max volume
                    self.write_reg(self.REG_KS_AR + op_off, 0x1F)  # KS=0, AR=31 (max)
                    self.write_reg(self.REG_AMS_D1R + op_off, 0x00)  # AMS=0, D1R=0
                    self.write_reg(self.REG_DT2_D2R + op_off, 0x00)  # DT2=0, D2R=0
                    # D1L=0 (sustain at full level), RR=0 (no release for sustain)
                    self.write_reg(self.REG_D1L_RR + op_off, 0x00)
                else:
                    # Operators 1-3 - silent but properly initialized
                    self.write_reg(self.REG_DT1_MUL + op_off, 0x01)  # DT1=0, MUL=1
                    self.write_reg(self.REG_TL + op_off, 0x7F)  # Max attenuation
                    self.write_reg(self.REG_KS_AR + op_off, 0x1F)  # KS=0, AR=31
                    self.write_reg(self.REG_AMS_D1R + op_off, 0x00)  # AMS=0, D1R=0
                    self.write_reg(self.REG_DT2_D2R + op_off, 0x00)  # DT2=0, D2R=0
                    self.write_reg(self.REG_D1L_RR + op_off, 0x00)  # D1L=0, RR=0

            # Initial key code (A4 = 440 Hz approximately)
            self.write_reg(self.REG_KC + ch, 0x4A)  # Octave 4, note A
            self.write_reg(self.REG_KF + ch, 0x00)

        # Silence unused channels
        for ch in range(num_channels, 8):
            for op in range(4):
                op_off = ch + op * 8
                self.write_reg(self.REG_TL + op_off, 0x7F)

    def enable_csm_mode(self, period_samples: int) -> None:
        """Enable CSM mode with Timer A.

        Timer A overflow triggers key-on for all 8 channels simultaneously.
        """
        # Set up Timer A period
        self._setup_timer_a(period_samples)
        # Enable CSM mode, Timer A IRQ, and load/start Timer A
        # Reset Timer A flag first, then enable CSM + load Timer A
        self.write_reg(
            self.REG_TIMER_CTRL,
            self.TIMER_CSM_ENABLE
            | self.TIMER_A_RESET
            | self.TIMER_A_IRQ_ENABLE
            | self.TIMER_A_LOAD,
        )

    def disable_csm_mode(self) -> None:
        """Disable CSM mode and stop Timer A."""
        # Reset timer flags and disable CSM
        self.write_reg(self.REG_TIMER_CTRL, self.TIMER_A_RESET | self.TIMER_B_RESET)

    def key_on(self, num_channels: int) -> None:
        """Trigger key-on for all active channels."""
        for ch in range(num_channels):
            # Key-on with all 4 operators enabled (bits 6-3 = 0xF)
            # 0xF8 = 11111000 = all 4 operators (bits 6,5,4,3) + channel in bits 2-0
            self.write_reg(self.REG_KEY_ON, 0xF8 | ch)

    def convert_frequency(self, freq: float) -> Tuple[int, int]:
        """Convert frequency to KC and KF values."""
        if freq <= 0:
            return (0, 0)

        ref_freq = 277.18  # C#4
        ref_octave = 4

        semitones_from_ref = 12 * np.log2(freq / ref_freq)
        total_semitones = semitones_from_ref
        octave = ref_octave + int(total_semitones // 12)
        semitone_in_octave = total_semitones % 12

        if semitone_in_octave < 0:
            semitone_in_octave += 12
            octave -= 1

        octave = max(0, min(7, octave))
        semitone_int = int(semitone_in_octave) % 12
        note = self.NOTE_MAP[semitone_int]

        frac = semitone_in_octave - int(semitone_in_octave)
        kf = int(frac * 64) & 0x3F
        kc = ((octave & 0x07) << 4) | (note & 0x0F)

        return (kc, kf)

    def update_frame(self, channel: int, freq_data: Tuple[int, int], tl: int) -> None:
        kc, kf = freq_data
        self.write_reg(self.REG_KC + channel, kc)
        self.write_reg(self.REG_KF + channel, kf << 2)
        self.write_reg(self.REG_TL + channel, tl)

    def key_off(self, num_channels: int) -> None:
        """Disable CSM mode and turn off all active channels."""
        self.disable_csm_mode()
        for ch in range(num_channels):
            # Key-off: channel number with no operator bits set
            self.write_reg(self.REG_KEY_ON, 0x00 | ch)


# =============================================================================
# OPN Base Class (YM2203/YM2608) with CSM Mode
# =============================================================================
class OPNChip(FMChip):
    """Base class for OPN-family chips (YM2203, YM2608) with true CSM mode support.

    CSM mode on OPN:
    - Controlled by bits 7-6 of register 0x27 (value 0x80 = CSM mode)
    - Timer A overflow triggers key-on for Channel 3 ONLY
    - CSM mode also enables per-operator frequency control for Channel 3
    - Each of the 4 operators can have independent F-Number
    - Timer A is 10-bit (registers 0x24 and 0x25)
    - Timer period: (1024 - TA) * 72 / clock

    Per-operator frequency registers for Channel 3 (extended mode):
    - Operator 1: 0xA9 (F-Num low), 0xAD (F-Num high + block)
    - Operator 2: 0xAA (F-Num low), 0xAE (F-Num high + block)
    - Operator 3: 0xA8 (F-Num low), 0xAC (F-Num high + block)
    - Operator 4: 0xA2 (F-Num low), 0xA6 (F-Num high + block) [base ch3 regs]
    """

    # Register definitions
    REG_KEY_ON = 0x28
    REG_TIMER_A_HI = 0x24  # Timer A bits 9-2
    REG_TIMER_A_LO = 0x25  # Timer A bits 1-0
    REG_TIMER_B = 0x26  # Timer B 8 bits
    REG_TIMER_CTRL = 0x27  # Timer control and CH3 mode
    REG_FNUM_LO = 0xA0
    REG_FNUM_HI_BLOCK = 0xA4
    REG_FB_CON = 0xB0
    REG_DT1_MUL = 0x30
    REG_TL = 0x40
    REG_KS_AR = 0x50
    REG_AM_DR = 0x60
    REG_SR = 0x70
    REG_SL_RR = 0x80

    # Timer control bits for register 0x27
    # Bit 6: CSM mode (Timer A overflow triggers key-on for CH3)
    # Bit 7: Per-operator frequency mode (each CH3 operator has independent F-Number)
    MODE_NORMAL = 0x00
    MODE_EXTENDED = 0x80  # Bit7: Per-operator frequency mode only
    MODE_CSM = 0x40  # Bit 6: CSM mode
    TIMER_B_RESET = 0x20  # Bit 5: Reset Timer B flag
    TIMER_A_RESET = 0x10  # Bit 4: Reset Timer A flag
    TIMER_B_ENABLE = 0x08  # Bit 3: Timer B overflow enable
    TIMER_A_ENABLE = 0x04  # Bit 2: Timer A overflow enable
    TIMER_B_LOAD = 0x02  # Bit 1: Load Timer B
    TIMER_A_LOAD = 0x01  # Bit 0: Load Timer A

    # Channel 3 extended mode per-operator frequency registers (low byte)
    # High byte register is always low + 4
    # OPN slot order is OP1, OP3, OP2, OP4 - freq registers follow same pattern!
    CH3_OP_FREQ_LO_REGS = [0xA9, 0xA8, 0xAA, 0xA2]

    PRESCALER = 72

    @property
    def max_channels(self) -> int:
        # In CSM mode, we use 4 operators of Channel 3 as "channels"
        return 4

    def _calculate_timer_a(self, period_samples: int) -> int:
        """Calculate Timer A value for a given period in samples at 44100 Hz.

        Timer A period formula: (1024 - TA) * 72 / clock
        Solving for TA: TA = 1024 - (period_seconds * clock / 72)
        """
        period_seconds = period_samples / VGM_SAMPLE_RATE
        timer_val = int(1024 - (period_seconds * self.clock / 72))
        return max(0, min(1023, timer_val))

    def _setup_timer_a(self, period_samples: int) -> None:
        """Set up Timer A for CSM mode."""
        timer_val = self._calculate_timer_a(period_samples)
        # Timer A high (bits 9-2)
        self.write_reg(self.REG_TIMER_A_HI, (timer_val >> 2) & 0xFF, port=0)
        # Timer A low (bits 1-0)
        self.write_reg(self.REG_TIMER_A_LO, timer_val & 0x03, port=0)

    def init_channels(self, num_channels: int) -> None:
        """Initialize Channel 3 for CSM mode with per-operator frequencies.

        In CSM mode, we only use Channel 3 (4 operators = 4 formants).
        """
        # Set up Channel 3 with algorithm 7 (all operators as carriers)
        self.write_reg(self.REG_FB_CON + 2, 0x07, port=0)

        # Set up all 4 operators of Channel 3
        for op in range(4):
            op_offset = 2 + 4 * op  # CH3 + slot offset
            self.write_reg(self.REG_DT1_MUL + op_offset, 0x01, port=0)  # MUL=1
            self.write_reg(self.REG_TL + op_offset, 0x7F, port=0)  # Start silent
            self.write_reg(self.REG_KS_AR + op_offset, 0x1F, port=0)  # Max AR
            self.write_reg(self.REG_AM_DR + op_offset, 0x00, port=0)  # No DR
            self.write_reg(self.REG_SR + op_offset, 0x00, port=0)  # No SR
            self.write_reg(
                self.REG_SL_RR + op_offset, 0x00, port=0
            )  # SL=0, RR=0 (sustain)

            # Initialize per-operator frequency (extended mode)
            # Write high first, then low (OPN latches high, triggers on low write)
            lo_reg = self.CH3_OP_FREQ_LO_REGS[op]
            self.write_reg(lo_reg + 4, 0x22, port=0)  # Block 4
            self.write_reg(lo_reg, 0x00, port=0)

        # Silence other channels (1, 2) on port 0 - note CH3 is index 2
        for ch in [0, 1]:
            for op in range(4):
                op_offset = ch + 4 * op
                self.write_reg(self.REG_TL + op_offset, 0x7F, port=0)

    def enable_csm_mode(self, period_samples: int) -> None:
        """Enable CSM mode and start Timer A."""
        # Set up Timer A period
        self._setup_timer_a(period_samples)
        # Enable CSM mode and load/enable Timer A
        self.write_reg(
            self.REG_TIMER_CTRL,
            self.MODE_CSM
            | self.TIMER_A_ENABLE
            | self.TIMER_A_LOAD
            | self.TIMER_A_RESET,
            port=0,
        )

    def disable_csm_mode(self) -> None:
        """Disable CSM mode and stop Timer A."""
        # Return to normal mode, reset timer flags
        self.write_reg(
            self.REG_TIMER_CTRL,
            self.MODE_NORMAL | self.TIMER_A_RESET | self.TIMER_B_RESET,
            port=0,
        )

    def convert_frequency(self, freq: float) -> Tuple[int, int]:
        """Convert frequency to F-Number and Block (11-bit F-Number for OPN)."""
        for block in range(8):
            fnum = int(freq * self.PRESCALER * (1 << (21 - block)) / self.clock)
            if 1 <= fnum <= 2047:
                return fnum, block

        return (2047, 7)

    def update_frame(self, channel: int, freq_data: Tuple[int, int], tl: int) -> None:
        """Update operator frequency and amplitude for CSM mode.

        In CSM mode, 'channel' is actually the operator index (0-3) of Channel 3.
        """
        fnum, block = freq_data
        op = channel  # In CSM mode, we use operator index

        # Get per-operator frequency register (high = low + 4)
        lo_reg = self.CH3_OP_FREQ_LO_REGS[op]

        # Write high first, then low (OPN latches high, triggers on low write)
        self.write_reg(
            lo_reg + 4,
            ((block & 0x07) << 3) | ((fnum >> 8) & 0x07),
            port=0,
        )
        self.write_reg(lo_reg, fnum & 0xFF, port=0)

        # Update TL for this operator (CH3 = channel index 2)
        op_offset = 2 + 4 * op  # CH3 + slot offset
        self.write_reg(self.REG_TL + op_offset, tl, port=0)

    def key_on(self, num_channels: int) -> None:
        """Manual key-on for Channel 3 (CSM mode handles this automatically via timer)."""
        # Key-on Channel 3 with all 4 operators enabled
        # In CSM mode, this is triggered automatically by Timer A overflow
        self.write_reg(self.REG_KEY_ON, 0xF0 | 2, port=0)  # All ops, CH3 = 2

    def key_off(self, num_channels: int) -> None:
        """Disable CSM mode and manually key off Channel 3."""
        self.disable_csm_mode()
        # Key off Channel 3 (all operators)
        self.write_reg(self.REG_KEY_ON, 0x00 | 2, port=0)  # CH3 = 2


class YM2203Chip(OPNChip):
    """YM2203 (OPN) FM sound chip with CSM mode support.

    In CSM mode, only Channel 3 is used (4 operators = 4 formants).
    """

    @property
    def name(self) -> str:
        return "YM2203"

    @property
    def clock(self) -> int:
        return 3579545

    @property
    def vgm_header_offset(self) -> int:
        return 0x44

    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        self.vgm.write_command(VGM_CMD_YM2203, reg, data)


class YM2608Chip(OPNChip):
    """YM2608 (OPNA) FM sound chip with CSM mode support.

    In CSM mode, only Channel 3 is used (4 operators = 4 formants).
    The higher clock rate affects timer calculations.
    """

    PRESCALER = 144  # YM2608 uses 6*24=144 prescaler

    @property
    def name(self) -> str:
        return "YM2608"

    @property
    def clock(self) -> int:
        return 7987200

    @property
    def vgm_header_offset(self) -> int:
        return 0x48

    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        opcode = VGM_CMD_YM2608_P0 if port == 0 else VGM_CMD_YM2608_P1
        self.vgm.write_command(opcode, reg, data)


# =============================================================================
# OPL Base Class (YM3812/Y8950) with CSW Mode
# =============================================================================
class OPLChip(FMChip):
    """Base class for OPL-family chips (YM3812, Y8950) with true CSW mode support.

    CSW (Composite Sine Wave) mode on OPL2:
    - Controlled by bit 7 of register 0x08
    - Timer 1 overflow triggers key-on for ALL 9 channels simultaneously
    - Timer 1 is 8-bit with 80µs resolution
    - Timer 1 period: (256 - T1) * 80µs
    """

    def __init__(self, vgm: VGMWriter):
        super().__init__(vgm)

    # Register definitions
    REG_TEST = 0x01  # Test/Waveform Select Enable
    REG_TIMER1 = 0x02  # Timer 1 value (80µs resolution)
    REG_TIMER2 = 0x03  # Timer 2 value (320µs resolution)
    REG_TIMER_CTRL = 0x04  # Timer control
    REG_CSW_NOTESEL = 0x08  # CSW mode and NOTE-SEL
    REG_AM_VIB_EG_KSR_MUL = 0x20
    REG_KSL_TL = 0x40
    REG_AR_DR = 0x60
    REG_SL_RR = 0x80
    REG_FNUM_LO = 0xA0
    REG_KEY_BLOCK_FNUM_HI = 0xB0
    REG_FB_CON = 0xC0

    # Timer control bits (register 0x04)
    TIMER_IRQ_RESET = 0x80  # Bit 7: Reset IRQ flags
    TIMER1_MASK = 0x40  # Bit 6: Timer 1 mask (don't set IRQ flag)
    TIMER2_MASK = 0x20  # Bit 5: Timer 2 mask
    TIMER2_START = 0x02  # Bit 1: Timer 2 start
    TIMER1_START = 0x01  # Bit 0: Timer 1 start

    # CSW/NOTE-SEL bits (register 0x08)
    CSW_ENABLE = 0x80  # Bit 7: CSW mode enable
    NOTE_SEL = 0x40  # Bit 6: NOTE-SEL

    # Modulator register offset for each channel.
    # Carrier offset is always modulator + 3.
    SLOT_MAP = [0, 1, 2, 8, 9, 10, 16, 17, 18]

    @property
    def max_channels(self) -> int:
        return 9

    def _calculate_timer1(self, period_samples: int) -> int:
        """Calculate Timer 1 value for a given period in samples at 44100 Hz.

        Timer 1 period formula: (256 - T1) * 80µs
        Solving for T1: T1 = 256 - (period_seconds / 80e-6)
        """
        period_seconds = period_samples / VGM_SAMPLE_RATE
        timer_val = int(256 - (period_seconds / 80e-6))
        return max(0, min(255, timer_val))

    def _setup_timer1(self, period_samples: int) -> None:
        """Set up Timer 1 for CSW mode."""
        timer_val = self._calculate_timer1(period_samples)
        self.write_reg(self.REG_TIMER1, timer_val)

    def init_channels(self, num_channels: int) -> None:
        num_channels = min(num_channels, self.max_channels)

        for ch in range(num_channels):
            mod_offset = self.SLOT_MAP[ch]
            car_offset = mod_offset + 3

            # Modulator settings (silent in additive mode)
            self.write_reg(self.REG_AM_VIB_EG_KSR_MUL + mod_offset, 0x01)  # MUL=1
            self.write_reg(self.REG_KSL_TL + mod_offset, 0x3F)  # Max attenuation
            self.write_reg(self.REG_AR_DR + mod_offset, 0xF0)  # Max AR, DR=0
            self.write_reg(self.REG_SL_RR + mod_offset, 0x00)  # SL=0, RR=0

            # Carrier settings (audible)
            self.write_reg(self.REG_AM_VIB_EG_KSR_MUL + car_offset, 0x01)  # MUL=1
            self.write_reg(self.REG_KSL_TL + car_offset, 0x3F)  # Start silent
            self.write_reg(self.REG_AR_DR + car_offset, 0xF0)  # Max AR, DR=0
            self.write_reg(self.REG_SL_RR + car_offset, 0x00)  # SL=0, RR=0 (sustain)

            # FB/CON: No feedback, additive synthesis (CON=1)
            self.write_reg(self.REG_FB_CON + ch, 0x01)

            # Initial frequency (key-on bit NOT set - CSW handles it)
            self.write_reg(self.REG_FNUM_LO + ch, 0x00)
            self.write_reg(self.REG_KEY_BLOCK_FNUM_HI + ch, 0x00)

        # Silence unused channels
        for ch in range(num_channels, 9):
            mod_offset = self.SLOT_MAP[ch]
            car_offset = mod_offset + 3
            self.write_reg(self.REG_KSL_TL + mod_offset, 0x3F)
            self.write_reg(self.REG_KSL_TL + car_offset, 0x3F)

    def enable_csm_mode(self, period_samples: int) -> None:
        """Enable CSW mode and start Timer 1."""
        # Set up Timer 1 period
        self._setup_timer1(period_samples)
        # Start Timer 1
        self.write_reg(self.REG_TIMER_CTRL, self.TIMER_IRQ_RESET)  # Reset flags first
        self.write_reg(self.REG_TIMER_CTRL, self.TIMER1_START)
        # Enable CSW mode
        self.write_reg(self.REG_CSW_NOTESEL, self.CSW_ENABLE)

    def disable_csm_mode(self) -> None:
        """Disable CSW mode and stop Timer 1."""
        # Disable CSW mode
        self.write_reg(self.REG_CSW_NOTESEL, 0x00)
        # Stop Timer 1 and reset flags
        self.write_reg(self.REG_TIMER_CTRL, self.TIMER_IRQ_RESET)

    def convert_frequency(self, freq: float) -> Tuple[int, int]:
        """Convert frequency to F-Number and Block (10-bit F-Number for OPL)."""
        for block in range(8):
            fnum = int(freq * 72 * (1 << (20 - block)) / self.clock)
            if 1 <= fnum <= 1023:
                return (fnum, block)

        return (1023, 7)

    def convert_amplitude(self, amp: float, max_amp: float) -> int:
        """OPL uses 6-bit TL (0-63)."""
        tl_7bit = super().convert_amplitude(amp, max_amp)
        return (tl_7bit >> 1) & 0x3F

    def update_frame(self, channel: int, freq_data: Tuple[int, int], tl: int) -> None:
        """Update channel frequency and amplitude.

        Always includes KEY_ON bit to maintain note sustain. Writing
        without KEY_ON would trigger release phase with fast RR setting.
        """
        fnum, block = freq_data
        car_offset = self.SLOT_MAP[channel] + 3

        # Write frequency WITH KEY_ON bit (0x20) to maintain sustain
        self.write_reg(self.REG_FNUM_LO + channel, fnum & 0xFF)
        self.write_reg(
            self.REG_KEY_BLOCK_FNUM_HI + channel,
            0x20 | ((block & 0x07) << 2) | ((fnum >> 8) & 0x03),
        )
        self.write_reg(self.REG_KSL_TL + car_offset, tl)

    def key_on(self, num_channels: int) -> None:
        """Manual key-on for all channels.

        Since update_frame already maintains KEY_ON, this is a no-op.
        """
        # KEY_ON is already maintained by update_frame

    def key_off(self, num_channels: int) -> None:
        """Disable CSW mode and clear key-on for all channels."""
        self.disable_csm_mode()
        for ch in range(num_channels):
            self.write_reg(self.REG_KEY_BLOCK_FNUM_HI + ch, 0x00)


class YM3526Chip(OPLChip):
    """YM3526 (OPL) FM sound chip."""

    @property
    def name(self) -> str:
        return "YM3526"

    @property
    def clock(self) -> int:
        return 3579545

    @property
    def vgm_header_offset(self) -> int:
        return 0x4C

    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        self.vgm.write_command(VGM_CMD_YM3526, reg, data)


class YM3812Chip(OPLChip):
    """YM3812 (OPL2) FM sound chip."""

    @property
    def name(self) -> str:
        return "YM3812"

    @property
    def clock(self) -> int:
        return 3579545

    @property
    def vgm_header_offset(self) -> int:
        return 0x50

    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        self.vgm.write_command(VGM_CMD_YM3812, reg, data)


class Y8950Chip(OPLChip):
    """Y8950 (MSX-Audio) FM sound chip."""

    @property
    def name(self) -> str:
        return "Y8950"

    @property
    def clock(self) -> int:
        return 3579545

    @property
    def vgm_header_offset(self) -> int:
        return 0x58  # VGM spec: Y8950 at offset 0x58

    def write_reg(self, reg: int, data: int, port: int = 0) -> None:
        self.vgm.write_command(VGM_CMD_Y8950, reg, data)


# =============================================================================
# Chip Factory
# =============================================================================
CHIP_CLASSES: dict[str, Callable[[VGMWriter], FMChip]] = {
    "ym2151": YM2151Chip,
    "ym2203": YM2203Chip,
    "ym2608": YM2608Chip,
    "ym3526": YM3526Chip,
    "ym3812": YM3812Chip,
    "y8950": Y8950Chip,
}


def create_chip(chip_name: str, vgm: VGMWriter) -> FMChip:
    """Factory function to create chip instances."""
    chip_name = chip_name.lower()
    if chip_name not in CHIP_CLASSES:
        raise ValueError(
            f"Unknown chip: {chip_name}. Supported: {list(CHIP_CLASSES.keys())}"
        )
    return CHIP_CLASSES[chip_name](vgm)


# =============================================================================
# DSP Helper Functions
# =============================================================================
def stft(
    waveform: Annotated[NDArray[np.floating], "(n_samples,)"],
    n_fft: int,
    hop_length: int,
) -> Annotated[NDArray[np.float64], "(n_fft//2+1, num_frames)"]:
    """Compute Short-Time Fourier Transform using numpy."""
    # Generate Hann window
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)

    # Pad waveform to ensure we have complete frames
    pad_length = n_fft // 2
    waveform_padded = np.concatenate(
        [np.zeros(pad_length), waveform, np.zeros(pad_length)]
    )

    # Calculate number of frames
    num_frames = 1 + (len(waveform_padded) - n_fft) // hop_length

    # Pre-allocate output (only positive frequencies)
    n_freqs = n_fft // 2 + 1
    S = np.zeros((n_freqs, num_frames), dtype=np.float64)

    # Compute STFT frame by frame
    for i in range(num_frames):
        start = i * hop_length
        frame = waveform_padded[start : start + n_fft] * window
        spectrum = np.fft.rfft(frame)
        S[:, i] = np.abs(spectrum)

    return S


def resample(
    waveform: Annotated[NDArray[np.floating], "(n_samples,)"], target_length: int
) -> Annotated[NDArray[np.floating], "(target_length,)"]:
    """Resample waveform to target length using linear interpolation."""
    if len(waveform) == target_length:
        return waveform

    x_old = np.linspace(0, 1, len(waveform))
    x_new = np.linspace(0, 1, target_length)
    return np.interp(x_new, x_old, waveform)


# =============================================================================
# CSM Data Extraction
# =============================================================================
def extract_csm_data(
    waveform: Annotated[NDArray[np.floating], "(n_samples,)"], sample_rate: int = 44100
) -> Annotated[NDArray[np.float64], "(num_frames, 4, 2)"]:
    """Extract CSM data (4 formant frequencies + amplitudes) from audio."""
    # STFT parameters
    n_fft = int(sample_rate * 40e-3)  # 40ms window
    hop_length = n_fft // 4  # 10ms hop (stride)

    # Perform STFT
    S = stft(waveform, n_fft, hop_length)

    # Frequency bins
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
    num_frames = S.shape[1]

    # Pre-allocate output array
    csm_data = np.zeros((num_frames, 4, 2), dtype=np.float64)

    # Smoothing filter
    filt = np.array([1, 2, 4, 2, 1], dtype=np.float64)
    filt_sum = np.sum(filt)
    filt_len = len(filt)
    norm_factor = filt_sum * filt_len

    # Process all frames
    for i in range(num_frames):
        item = S[:, i]

        # Inline smoothing for performance
        padded = np.concatenate([[item[1], item[2]], item, [item[-2], item[-3]]])
        item_smooth = np.convolve(padded, filt, mode="valid") / norm_factor

        # Inline peak finding
        if len(item_smooth) >= 3:
            is_peak = (item_smooth[1:-1] > item_smooth[:-2]) & (
                item_smooth[1:-1] > item_smooth[2:]
            )
            peak_indices = np.where(is_peak)[0] + 1

            if len(peak_indices) > 0:
                peak_vals = item_smooth[peak_indices]
                # Get top 4 peaks by amplitude
                if len(peak_indices) > 4:
                    top4_idx = np.argpartition(peak_vals, -4)[-4:]
                    top4_idx = top4_idx[np.argsort(peak_vals[top4_idx])[::-1]]
                else:
                    top4_idx = np.argsort(peak_vals)[::-1]

                for ii, idx in enumerate(top4_idx[:4]):
                    data_idx = peak_indices[idx]
                    csm_data[i, ii, 0] = (
                        freqs[data_idx] if data_idx < len(freqs) else 0.0
                    )
                    csm_data[i, ii, 1] = item[data_idx]

    return csm_data


# =============================================================================
# Main Conversion Function
# =============================================================================
def load_wav(
    filename: str,
) -> Tuple[Annotated[NDArray[np.floating], "(n_samples,)"], int]:
    """Load a WAV file and return (waveform, sample_rate)."""
    with wave.open(filename, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    # Convert bytes to numpy array based on sample width
    if sample_width == 1:
        # 8-bit unsigned
        waveform = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64)
        waveform = (waveform - 128) / 128.0
    elif sample_width == 2:
        # 16-bit signed
        waveform = np.frombuffer(raw_data, dtype=np.int16).astype(np.float64)
        waveform = waveform / 32768.0
    elif sample_width == 3:
        # 24-bit signed (needs special handling)
        n_samples = len(raw_data) // 3
        waveform = np.zeros(n_samples, dtype=np.float64)
        for i in range(n_samples):
            # Little-endian 24-bit to signed int
            val = (
                raw_data[i * 3]
                | (raw_data[i * 3 + 1] << 8)
                | (raw_data[i * 3 + 2] << 16)
            )
            if val >= 0x800000:
                val -= 0x1000000
            waveform[i] = val / 8388608.0
    elif sample_width == 4:
        # 32-bit signed
        waveform = np.frombuffer(raw_data, dtype=np.int32).astype(np.float64)
        waveform = waveform / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Reshape for multi-channel and convert to mono
    if n_channels > 1:
        waveform = waveform.reshape(-1, n_channels)
        waveform = np.mean(waveform, axis=1)

    return waveform, sample_rate


def convert_wav_to_vgm(
    input_wav: str, output_vgm: str, chip_name: str = "ym2151"
) -> None:
    """Convert a WAV file to VGM format using true CSM hardware mode.

    CSM mode uses the chip's built-in timer to automatically trigger
    key-on events at a precise rate, creating the characteristic
    "grainy" speech synthesis effect.
    """
    print(f"Loading: {input_wav}")

    # Load audio using built-in wave module
    waveform, sample_rate = load_wav(input_wav)

    # Resample to VGM sample rate if needed
    if sample_rate != VGM_SAMPLE_RATE:
        num_samples = int(len(waveform) * VGM_SAMPLE_RATE / sample_rate)
        waveform = resample(waveform, num_samples)

    duration = len(waveform) / VGM_SAMPLE_RATE
    print(f"Duration: {duration:.2f} seconds")

    # Create VGM writer and chip
    vgm = VGMWriter()
    chip = create_chip(chip_name, vgm)
    print(f"Target chip: {chip.name}")

    # Determine number of channels/operators for formants
    num_channels = min(4, chip.max_channels)

    # Extract CSM data (formant frequencies and amplitudes)
    print("Extracting formants...")
    csm_data = extract_csm_data(waveform, VGM_SAMPLE_RATE)
    num_frames = csm_data.shape[0]
    print(f"Frames: {num_frames}")

    # Find maximum amplitude for normalization
    max_amp = np.max(csm_data[:, :, 1])
    if max_amp == 0:
        print("Warning: No audio detected")
        max_amp = np.float64(1.0)

    # Samples per CSM frame (10ms at 44100 Hz = 441 samples)
    samples_per_frame = int(VGM_SAMPLE_RATE * 10e-3)  # 441 samples

    # Initialize chip channels/operators
    print("Generating VGM data with CSM mode...")
    chip.init_channels(num_channels)

    # Enable CSM mode with timer synchronized to frame rate
    # The timer overflow rate should match our parameter update rate
    chip.enable_csm_mode(samples_per_frame)

    # Process each frame
    for frame_idx in range(num_frames):
        # Update frequency and amplitude for each formant channel/operator
        for ch in range(num_channels):
            freq = csm_data[frame_idx, ch, 0]
            amp = csm_data[frame_idx, ch, 1]

            freq_data = chip.convert_frequency(freq)
            tl = chip.convert_amplitude(amp, max_amp)
            chip.update_frame(ch, freq_data, tl)

        # Manual key-on for each frame (CSM mode timer may not work in all players)
        chip.key_on(num_channels)

        # Wait for next frame
        vgm.write_wait(samples_per_frame)

    # Disable CSM mode and key off
    chip.key_off(num_channels)
    vgm.write_end()

    # Save
    vgm.save(output_vgm)
    print(f"Saved: {output_vgm}")
    print(
        f"Total samples: {vgm.total_samples} ({vgm.total_samples / VGM_SAMPLE_RATE:.2f} sec)"
    )


# =============================================================================
# Test Mode - Generate frequency sweep for debugging
# =============================================================================
def generate_test_vgm(output_vgm: str, chip_name: str = "ym2203") -> None:
    """Generate a test VGM with different frequencies for each operator.

    Test mode: Each operator plays a DIFFERENT constant frequency to verify
    per-operator frequency mode works.
    """
    print(f"Generating test VGM for {chip_name}")

    vgm = VGMWriter()
    chip = create_chip(chip_name, vgm)

    num_channels = min(4, chip.max_channels)
    samples_per_frame = int(VGM_SAMPLE_RATE * 10e-3)  # 10ms per frame

    # Test: each operator plays a different constant frequency
    # If per-op freq works, we hear a chord. If not, we hear only one freq.
    test_freqs = [220.0, 330.0, 440.0, 550.0]  # A3, E4, A4, C#5 - should make a chord
    duration = 2.0  # seconds
    total_frames = int(duration / 0.01)

    # Initialize
    chip.init_channels(num_channels)
    chip.enable_csm_mode(samples_per_frame)

    print(f"Operators: {num_channels}")
    print("Test: Each operator plays a different constant frequency")
    for ch in range(num_channels):
        print(f"  Op{ch}: {test_freqs[ch]:.0f} Hz")
    print(f"Duration: {duration}s")
    print("If per-op freq works: hear a chord (multiple freqs)")
    print("If per-op freq fails: hear only one frequency")

    # Generate frames
    for frame_idx in range(total_frames):
        for ch in range(num_channels):
            freq = test_freqs[ch]
            tl = 0  # All operators audible

            freq_data = chip.convert_frequency(freq)
            chip.update_frame(ch, freq_data, tl)

        chip.key_on(num_channels)
        vgm.write_wait(samples_per_frame)

    chip.key_off(num_channels)
    vgm.write_end()

    vgm.save(output_vgm)
    print(f"Saved: {output_vgm}")


# =============================================================================
# Main Entry Point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert WAV file to VGM using FM chip CSM voice synthesis"
    )
    parser.add_argument(
        "output",
        nargs="?",
        help="Output VGM file (default: input.vgm or test_CHIP.vgm)",
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Input WAV file (required unless --test is specified)",
    )
    parser.add_argument(
        "--chip",
        "-c",
        choices=list(CHIP_CLASSES.keys()),
        default="ym2151",
        help="Target FM chip (default: ym2151)",
    )
    parser.add_argument(
        "--test",
        "-t",
        action="store_true",
        help="Generate test VGM with frequency sweeps (no input file needed)",
    )

    args = parser.parse_args()

    if args.test:
        # Test mode - generate frequency sweep
        output_vgm = args.output or f"test_{args.chip}.vgm"
        generate_test_vgm(output_vgm, chip_name=args.chip)
        return

    if not args.input:
        parser.error("--input is required (unless using --test mode)")

    input_wav = args.input
    if args.output:
        output_vgm = args.output
    else:
        base, _ = os.path.splitext(input_wav)
        output_vgm = base + ".vgm"

    if not os.path.exists(input_wav):
        print(f"Error: Input file not found: {input_wav}")
        sys.exit(1)

    convert_wav_to_vgm(input_wav, output_vgm, chip_name=args.chip)


if __name__ == "__main__":
    main()
