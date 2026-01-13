#!/usr/bin/env python3
"""
VGM file renderer using ymfm-py.

This is a Python port of the vgmrender example from ymfm.
It renders VGM (Video Game Music) files to WAV format.

Original C++ implementation:
    Copyright (c) 2021, Aaron Giles
    https://github.com/aaronsgiles/ymfm
    BSD 3-Clause License

Python port:
    Part of ymfm-py - Python bindings for ymfm

Usage:
    python vgmrender.py input.vgm -o output.wav [-r 44100]
"""

import argparse
import gzip
import struct
import sys
import wave
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

import ymfm


class ChipFactory(Protocol):
    """Protocol for chip class constructors."""

    def __call__(
        self, clock: int, interface: Optional[ymfm.ChipInterface] = None
    ) -> ymfm.Chip: ...


class ChipType(IntEnum):
    """Enumeration of supported chip types."""

    YM2149 = 0
    YM2151 = 1
    YM2203 = 2
    YM2413 = 3
    YM2608 = 4
    YM2610 = 5
    YM2612 = 6
    YM3526 = 7
    Y8950 = 8
    YM3812 = 9
    YMF262 = 10


# Mapping from ChipType to pymfm chip class
CHIP_CLASSES: dict[ChipType, ChipFactory] = {
    ChipType.YM2149: ymfm.YM2149,
    ChipType.YM2151: ymfm.YM2151,
    ChipType.YM2203: ymfm.YM2203,
    ChipType.YM2413: ymfm.YM2413,
    ChipType.YM2608: ymfm.YM2608,
    ChipType.YM2610: ymfm.YM2610,
    ChipType.YM2612: ymfm.YM2612,
    ChipType.YM3526: ymfm.YM3526,
    ChipType.Y8950: ymfm.Y8950,
    ChipType.YM3812: ymfm.YM3812,
    ChipType.YMF262: ymfm.YMF262,
}


@dataclass
class VgmChip:
    """Wrapper for a VGM chip instance."""

    chip_type: ChipType
    chip: ymfm.Chip
    clock: int
    name: str
    data: dict = field(
        default_factory=lambda: {
            ymfm.AccessClass.IO: bytearray(),
            ymfm.AccessClass.ADPCM_A: bytearray(),
            ymfm.AccessClass.ADPCM_B: bytearray(),
            ymfm.AccessClass.PCM: bytearray(),
        }
    )
    pcm_offset: int = 0
    queue: list = field(default_factory=list)
    step: int = 0
    pos: int = 0
    last_output: memoryview | np.ndarray[tuple[int, int], np.dtype[np.int32]] | None = (
        None
    )

    def __post_init__(self) -> None:
        sample_rate = self.chip.sample_rate
        # step is in 32.32 fixed point format
        self.step = (1 << 32) // sample_rate
        self.pos = 0
        self.last_output = self.chip.generate(1)

    @property
    def sample_rate(self) -> int:
        return self.chip.sample_rate

    def write(self, reg: int, data: int):
        """Queue a register write."""
        self.queue.append((reg, data))

    def write_data(self, access_type: ymfm.AccessClass, base: int, data: bytes):
        """Write data to the chip's memory buffer."""
        buf = self.data[access_type]
        end = base + len(data)
        if end > len(buf):
            buf.extend(bytes(end - len(buf)))
        buf[base : base + len(data)] = data

    def seek_pcm(self, pos: int):
        """Seek within the PCM stream."""
        self.pcm_offset = pos

    def read_pcm(self) -> int:
        """Read a byte from the PCM stream."""
        pcm = self.data[ymfm.AccessClass.PCM]
        if self.pcm_offset < len(pcm):
            val = pcm[self.pcm_offset]
            self.pcm_offset += 1
            return val
        return 0

    def generate(self, output_start: int, output_step: int) -> tuple:
        """Generate samples and return (left, right) output (per-sample version)."""
        addr1, addr2 = 0xFFFF, 0xFFFF
        data1, data2 = 0, 0

        # Process queued writes
        if self.queue:
            reg, val = self.queue.pop(0)
            addr1 = 0 + 2 * ((reg >> 8) & 3)
            data1 = reg & 0xFF
            addr2 = addr1 + 1
            data2 = val

        # Write to the chip
        if addr1 != 0xFFFF:
            self.chip.write(addr1, data1)
            self.chip.write(addr2, data2)

        # Generate at the appropriate sample rate
        while self.pos <= output_start:
            self.last_output = self.chip.generate(1)
            self.pos += self.step

        # Return stereo output based on chip type
        # last_output is a 2D memoryview with shape (num_samples, outputs)
        assert self.last_output is not None
        out = self.last_output
        outputs = self.chip.outputs

        if self.chip_type == ChipType.YM2149:
            # YM2149/AY-3-8910: 3 channels (A, B, C), mix to mono and duplicate
            total = sum(out[0, i % outputs] for i in range(3))
            return total, total
        elif self.chip_type == ChipType.YM2203:
            # YM2203: Mix all outputs to mono, duplicate to stereo
            total = sum(out[0, i % outputs] for i in range(4))
            return total, total
        elif self.chip_type in (ChipType.YM2608, ChipType.YM2610):
            # YM2608/YM2610: out0+out2 left, out1+out2 right
            out0 = out[0, 0]
            out1 = out[0, 1 % outputs]
            out2 = out[0, 2 % outputs]
            return out0 + out2, out1 + out2
        elif outputs == 1:
            # Mono chip: duplicate to stereo
            return out[0, 0], out[0, 0]
        else:
            # Stereo chip
            return out[0, 0], out[0, 1 % outputs]

    def has_pending_writes(self) -> bool:
        """Check if there are pending register writes."""
        return len(self.queue) > 0

    def generate_batch_no_writes(
        self, count: int, output_step: int, output_start: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a batch of output samples when no writes are pending.

        This is an optimized version that generates all native samples at once.
        Only call this when self.queue is empty!

        Args:
            count: Number of output samples to generate
            output_step: Step size in fixed-point format
            output_start: Starting position in fixed-point format

        Returns:
            Tuple of (left_samples, right_samples) as numpy arrays
        """
        if count == 0:
            return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

        outputs = self.chip.outputs

        # Calculate end position for the last output sample
        output_end = output_start + (count - 1) * output_step

        # Save starting position and convert last_output before generating
        start_pos = self.pos
        if isinstance(self.last_output, np.ndarray):
            last_output_row = self.last_output.reshape(1, outputs)
        else:
            # Convert 2D memoryview (1, outputs) to numpy array
            last_output_row = np.asarray(self.last_output, dtype=np.int32)

        # Calculate how many native samples we need
        num_native = 0
        if self.pos <= output_end:
            num_native = (output_end - self.pos) // self.step + 1

        # Generate all native samples at once
        if num_native > 0:
            native_output = self.chip.generate(num_native)
            native_array = np.asarray(native_output, dtype=np.int32).reshape(
                num_native, outputs
            )
            self.pos += num_native * self.step
            self.last_output = native_array[-1:, :].copy()
        else:
            native_array = np.empty((0, outputs), dtype=np.int32)

        # Prepend last_output for index 0
        all_samples = (
            np.vstack([last_output_row, native_array])
            if num_native > 0
            else last_output_row
        )

        # Calculate sample indices for each output position
        output_positions = output_start + np.arange(count, dtype=np.int64) * output_step
        sample_indices = (output_positions - start_pos) // self.step + 1
        sample_indices = np.clip(sample_indices, 0, len(all_samples) - 1)

        # Select samples
        selected = all_samples[sample_indices]

        # Mix channels based on chip type
        if self.chip_type == ChipType.YM2149:
            # YM2149/AY-3-8910: 3 channels (A, B, C), mix to mono
            total = selected[:, : min(3, outputs)].sum(axis=1)
            return total, total.copy()
        elif self.chip_type == ChipType.YM2203:
            # YM2203: 4 outputs, mix to mono
            total = selected[:, : min(4, outputs)].sum(axis=1)
            return total, total.copy()
        elif self.chip_type in (ChipType.YM2608, ChipType.YM2610):
            # YM2608/YM2610: out0+out2 left, out1+out2 right
            return selected[:, 0] + selected[:, 2], selected[:, 1] + selected[:, 2]
        elif outputs == 1:
            # Mono: duplicate
            mono = selected[:, 0]
            return mono, mono.copy()
        else:
            # Stereo: direct output
            return selected[:, 0], selected[:, 1]


class VgmPlayer:
    """VGM file player."""

    def __init__(self, output_rate: int = 44100):
        self.output_rate = output_rate
        self.chips: list[VgmChip] = []

    def add_chip(self, clock: int, chip_type: ChipType, name: str):
        """Add one or two chips of the given type."""
        clock_val = clock & 0x3FFFFFFF
        num_chips = 2 if (clock & 0x40000000) else 1

        chip_class = CHIP_CLASSES.get(chip_type)
        if chip_class is None:
            print(f"Warning: {name} not supported", file=sys.stderr)
            return

        print(f"Adding {'2 x ' if num_chips == 2 else ''}{name} @ {clock_val}Hz")

        for i in range(num_chips):
            chip_name = f"{name} #{i}" if num_chips == 2 else name
            # Use MemoryInterface for chips that need ADPCM data
            if chip_type in (ChipType.YM2608, ChipType.YM2610, ChipType.Y8950):
                interface = ymfm.MemoryInterface()
                chip = chip_class(clock=clock_val, interface=interface)
            else:
                chip = chip_class(clock=clock_val)
            chip.reset()
            self.chips.append(
                VgmChip(
                    chip_type=chip_type,
                    chip=chip,
                    clock=clock_val,
                    name=chip_name,
                )
            )

        # Load YM2608 ADPCM ROM if available
        if chip_type == ChipType.YM2608:
            rom_path = Path("ym2608_adpcm_rom.bin")
            if rom_path.exists():
                rom_data = rom_path.read_bytes()
                for vgm_chip in self.chips:
                    if vgm_chip.chip_type == chip_type:
                        vgm_chip.write_data(ymfm.AccessClass.ADPCM_A, 0, rom_data)
            else:
                print(
                    "Warning: YM2608 enabled but ym2608_adpcm_rom.bin not found",
                    file=sys.stderr,
                )

    def find_chip(self, chip_type: ChipType, index: int) -> Optional[VgmChip]:
        """Find a chip by type and index."""
        count = 0
        for chip in self.chips:
            if chip.chip_type == chip_type:
                if count == index:
                    return chip
                count += 1
        return None

    def write_chip(self, chip_type: ChipType, index: int, reg: int, data: int):
        """Write to a chip."""
        chip = self.find_chip(chip_type, index)
        if chip:
            chip.write(reg, data)

    def add_rom_data(
        self,
        chip_type: ChipType,
        access_type: ymfm.AccessClass,
        buffer: bytes,
        offset: int,
        size: int,
    ):
        """Add ROM data to chips of the given type."""
        _length = struct.unpack_from("<I", buffer, offset)[0]
        start = struct.unpack_from("<I", buffer, offset + 4)[0]
        data = buffer[offset + 8 : offset + 8 + size]

        for i in range(2):
            chip = self.find_chip(chip_type, i)
            if chip:
                chip.write_data(access_type, start, data)
                # Also update the MemoryInterface if present
                interface = chip.chip.interface
                if isinstance(interface, ymfm.MemoryInterface):
                    # Extend memory if needed
                    current = interface.get_memory()
                    if start + len(data) > len(current):
                        new_mem = bytearray(current)
                        new_mem.extend(bytes(start + len(data) - len(new_mem)))
                        new_mem[start : start + len(data)] = data
                        interface.set_memory(bytes(new_mem))
                    else:
                        new_mem = bytearray(current)
                        new_mem[start : start + len(data)] = data
                        interface.set_memory(bytes(new_mem))

    def parse_header(self, buffer: bytes) -> int:
        """Parse VGM header and create chips. Returns data start offset."""
        offset = 4  # Skip 'Vgm '

        # +04: Size
        _size = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4

        # +08: Version
        version = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version > 0x171:
            print(
                "Warning: version > 1.71 detected, some things may not work",
                file=sys.stderr,
            )

        # +0C: SN76489 clock (not supported)
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if clock != 0:
            print("Warning: SN76489 specified but not supported", file=sys.stderr)

        # +10: YM2413 clock
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if clock != 0:
            self.add_chip(clock, ChipType.YM2413, "YM2413")

        # +14: GD3 offset
        offset += 4
        # +18: Total samples
        offset += 4
        # +1C: Loop offset
        offset += 4
        # +20: Loop samples
        offset += 4
        # +24: Rate
        offset += 4
        # +28: SN76489 flags
        offset += 4

        # +2C: YM2612 clock
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x110 and clock != 0:
            self.add_chip(clock, ChipType.YM2612, "YM2612")

        # +30: YM2151 clock
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x110 and clock != 0:
            self.add_chip(clock, ChipType.YM2151, "YM2151")

        # +34: VGM data offset
        data_start = struct.unpack_from("<I", buffer, offset)[0]
        data_start += offset
        offset += 4
        if version < 0x150:
            data_start = 0x40

        # +38: Sega PCM clock (not supported)
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4

        # +3C: Sega PCM interface
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +40: RF5C68 clock (not supported)
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +44: YM2203 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.YM2203, "YM2203")

        # +48: YM2608 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.YM2608, "YM2608")

        # +4C: YM2610/B clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            # YM2610B if high bit set
            self.add_chip(
                clock & 0x7FFFFFFF,
                ChipType.YM2610,
                "YM2610B" if (clock & 0x80000000) else "YM2610",
            )

        # +50: YM3812 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.YM3812, "YM3812")

        # +54: YM3526 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.YM3526, "YM3526")

        # +58: Y8950 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.Y8950, "Y8950")

        # +5C: YMF262 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.YMF262, "YMF262")

        # +60: YMF278B clock (not supported)
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +64: YMF271 clock (not supported)
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +68: YMZ280B clock (not supported)
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +6C: RF5C164 clock (not supported)
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +70: PWM (not supported)
        if offset + 4 > data_start:
            return data_start
        offset += 4

        # +74: AY8910 clock
        if offset + 4 > data_start:
            return data_start
        clock = struct.unpack_from("<I", buffer, offset)[0]
        offset += 4
        if version >= 0x151 and clock != 0:
            self.add_chip(clock, ChipType.YM2149, "AY8910/YM2149")

        return data_start

    def generate_all(self, buffer: bytes, data_start: int) -> np.ndarray:
        """Process VGM commands and generate audio."""
        output_step = (1 << 32) // self.output_rate
        output_pos = 0
        offset = data_start
        done = False

        wav_buffer = []

        while not done and offset < len(buffer):
            delay = 0
            cmd = buffer[offset]
            offset += 1

            # YM2413
            if cmd in (0x51, 0xA1):
                self.write_chip(
                    ChipType.YM2413, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM2612 port 0
            elif cmd in (0x52, 0xA2):
                self.write_chip(
                    ChipType.YM2612, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM2612 port 1
            elif cmd in (0x53, 0xA3):
                self.write_chip(
                    ChipType.YM2612,
                    cmd >> 7,
                    buffer[offset] | 0x100,
                    buffer[offset + 1],
                )
                offset += 2

            # YM2151
            elif cmd in (0x54, 0xA4):
                self.write_chip(
                    ChipType.YM2151, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM2203
            elif cmd in (0x55, 0xA5):
                self.write_chip(
                    ChipType.YM2203, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM2608 port 0
            elif cmd in (0x56, 0xA6):
                self.write_chip(
                    ChipType.YM2608, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM2608 port 1
            elif cmd in (0x57, 0xA7):
                self.write_chip(
                    ChipType.YM2608,
                    cmd >> 7,
                    buffer[offset] | 0x100,
                    buffer[offset + 1],
                )
                offset += 2

            # YM2610 port 0
            elif cmd in (0x58, 0xA8):
                self.write_chip(
                    ChipType.YM2610, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM2610 port 1
            elif cmd in (0x59, 0xA9):
                self.write_chip(
                    ChipType.YM2610,
                    cmd >> 7,
                    buffer[offset] | 0x100,
                    buffer[offset + 1],
                )
                offset += 2

            # YM3812
            elif cmd in (0x5A, 0xAA):
                self.write_chip(
                    ChipType.YM3812, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YM3526
            elif cmd in (0x5B, 0xAB):
                self.write_chip(
                    ChipType.YM3526, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # Y8950
            elif cmd in (0x5C, 0xAC):
                self.write_chip(
                    ChipType.Y8950, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YMF262 port 0
            elif cmd in (0x5E, 0xAE):
                self.write_chip(
                    ChipType.YMF262, cmd >> 7, buffer[offset], buffer[offset + 1]
                )
                offset += 2

            # YMF262 port 1
            elif cmd in (0x5F, 0xAF):
                self.write_chip(
                    ChipType.YMF262,
                    cmd >> 7,
                    buffer[offset] | 0x100,
                    buffer[offset + 1],
                )
                offset += 2

            # Wait n samples
            elif cmd == 0x61:
                delay = buffer[offset] | (buffer[offset + 1] << 8)
                offset += 2

            # Wait 735 samples (60th of a second)
            elif cmd == 0x62:
                delay = 735

            # Wait 882 samples (50th of a second)
            elif cmd == 0x63:
                delay = 882

            # End of sound data
            elif cmd == 0x66:
                done = True

            # Data block
            elif cmd == 0x67:
                if buffer[offset] != 0x66:
                    offset += 1
                    continue
                offset += 1
                block_type = buffer[offset]
                offset += 1
                size = struct.unpack_from("<I", buffer, offset)[0]
                offset += 4
                local_offset = offset

                if block_type == 0x00:  # YM2612 PCM
                    chip = self.find_chip(ChipType.YM2612, 0)
                    if chip:
                        chip.write_data(
                            ymfm.AccessClass.PCM,
                            0,
                            buffer[local_offset : local_offset + size],
                        )
                elif block_type == 0x81:  # YM2608 DELTA-T
                    self.add_rom_data(
                        ChipType.YM2608,
                        ymfm.AccessClass.ADPCM_B,
                        buffer,
                        local_offset,
                        size - 8,
                    )
                elif block_type == 0x82:  # YM2610 ADPCM
                    self.add_rom_data(
                        ChipType.YM2610,
                        ymfm.AccessClass.ADPCM_A,
                        buffer,
                        local_offset,
                        size - 8,
                    )
                elif block_type == 0x83:  # YM2610 DELTA-T
                    self.add_rom_data(
                        ChipType.YM2610,
                        ymfm.AccessClass.ADPCM_B,
                        buffer,
                        local_offset,
                        size - 8,
                    )
                elif block_type == 0x88:  # Y8950 DELTA-T
                    self.add_rom_data(
                        ChipType.Y8950,
                        ymfm.AccessClass.ADPCM_B,
                        buffer,
                        local_offset,
                        size - 8,
                    )

                offset += size

            # Short delays (0x70-0x7F)
            elif 0x70 <= cmd <= 0x7F:
                delay = (cmd & 0x0F) + 1

            # YM2612 PCM + delay (0x80-0x8F)
            elif 0x80 <= cmd <= 0x8F:
                chip = self.find_chip(ChipType.YM2612, 0)
                if chip:
                    chip.write(0x2A, chip.read_pcm())
                delay = cmd & 0x0F

            # AY8910 / YM2149
            elif cmd == 0xA0:
                self.write_chip(ChipType.YM2149, 0, buffer[offset], buffer[offset + 1])
                offset += 2

            # PCM seek
            elif cmd == 0xE0:
                chip = self.find_chip(ChipType.YM2612, 0)
                pos = struct.unpack_from("<I", buffer, offset)[0]
                if chip:
                    chip.seek_pcm(pos)
                offset += 4

            # Ignored commands - consume appropriate bytes
            elif cmd in (
                0x30,
                0x31,
                0x32,
                0x33,
                0x34,
                0x35,
                0x36,
                0x37,
                0x38,
                0x39,
                0x3A,
                0x3B,
                0x3C,
                0x3D,
                0x3E,
                0x3F,
                0x4F,
                0x50,
            ):
                offset += 1
            elif cmd in (
                0x40,
                0x41,
                0x42,
                0x43,
                0x44,
                0x45,
                0x46,
                0x47,
                0x48,
                0x49,
                0x4A,
                0x4B,
                0x4C,
                0x4D,
                0x4E,
                0x5D,
                0xB0,
                0xB1,
                0xB2,
                0xB3,
                0xB4,
                0xB5,
                0xB6,
                0xB7,
                0xB8,
                0xB9,
                0xBA,
                0xBB,
                0xBC,
                0xBD,
                0xBE,
                0xBF,
            ):
                offset += 2
            elif cmd in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC4,
                0xC5,
                0xC6,
                0xC7,
                0xC8,
                0xC9,
                0xCA,
                0xCB,
                0xCC,
                0xCD,
                0xCE,
                0xCF,
                0xD0,
                0xD1,
                0xD2,
                0xD3,
                0xD4,
                0xD5,
                0xD6,
                0xD7,
                0xD8,
                0xD9,
                0xDA,
                0xDB,
                0xDC,
                0xDD,
                0xDE,
                0xDF,
            ):
                offset += 3
            elif cmd in (
                0xE1,
                0xE2,
                0xE3,
                0xE4,
                0xE5,
                0xE6,
                0xE7,
                0xE8,
                0xE9,
                0xEA,
                0xEB,
                0xEC,
                0xED,
                0xEE,
                0xEF,
                0xF0,
                0xF1,
                0xF2,
                0xF3,
                0xF4,
                0xF5,
                0xF6,
                0xF7,
                0xF8,
                0xF9,
                0xFA,
                0xFB,
                0xFC,
                0xFD,
                0xFE,
                0xFF,
            ):
                offset += 4

            # Handle delays - generate samples
            if delay > 0:
                # Phase 1: Process samples while any chip has pending writes
                samples_with_writes = []
                remaining = delay
                while remaining > 0 and any(
                    chip.has_pending_writes() for chip in self.chips
                ):
                    left, right = 0, 0
                    for chip in self.chips:
                        left_ch, right_ch = chip.generate(output_pos, output_step)
                        left += left_ch
                        right += right_ch
                    output_pos += output_step
                    samples_with_writes.append((left, right))
                    remaining -= 1

                if samples_with_writes:
                    wav_buffer.append(np.array(samples_with_writes, dtype=np.int32))

                # Phase 2: Batch generate remaining samples (no writes pending)
                if remaining > 0:
                    if len(self.chips) == 1:
                        # Single chip: no accumulation needed
                        left_total, right_total = self.chips[
                            0
                        ].generate_batch_no_writes(remaining, output_step, output_pos)
                    else:
                        # Multiple chips: accumulate
                        left_total = np.zeros(remaining, dtype=np.int32)
                        right_total = np.zeros(remaining, dtype=np.int32)
                        for chip in self.chips:
                            left_ch, right_ch = chip.generate_batch_no_writes(
                                remaining, output_step, output_pos
                            )
                            left_total += left_ch
                            right_total += right_ch
                    wav_buffer.append(np.column_stack((left_total, right_total)))
                    output_pos += remaining * output_step

        # Concatenate all chunks
        if wav_buffer:
            return np.vstack(wav_buffer)
        return np.array([], dtype=np.int32).reshape(0, 2)


def load_vgm(filename: str) -> bytes:
    """Load a VGM file, handling gzip compression."""
    path = Path(filename)
    data = path.read_bytes()

    # Check for gzip compression
    if len(data) >= 10 and data[0] == 0x1F and data[1] == 0x8B and data[2] == 0x08:
        data = gzip.decompress(data)

    # Verify VGM header
    if len(data) < 64 or data[0:4] != b"Vgm ":
        raise ValueError(f"File '{filename}' does not appear to be a valid VGM file")

    return data


def write_wav(filename: str, sample_rate: int, samples: np.ndarray):
    """Write samples to a WAV file with normalization."""
    if len(samples) == 0:
        print("Warning: No samples to write", file=sys.stderr)
        return

    # Normalize to 16-bit range
    max_val = np.max(np.abs(samples))
    if max_val == 0:
        print("Warning: WAV file will only contain silence", file=sys.stderr)
        max_val = 1

    # Scale to 16-bit with some headroom
    samples_16 = (samples * 26000 // max_val).astype(np.int16)

    # Write WAV file
    with wave.open(filename, "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples_16.tobytes())


def main():
    parser = argparse.ArgumentParser(description="Render VGM files to WAV")
    parser.add_argument("input", help="Input VGM file")
    parser.add_argument("-o", "--output", required=True, help="Output WAV file")
    parser.add_argument(
        "-r",
        "--rate",
        type=int,
        default=44100,
        help="Output sample rate (default: 44100)",
    )
    args = parser.parse_args()

    # Load VGM file
    try:
        vgm_data = load_vgm(args.input)
    except Exception as e:
        print(f"Error loading VGM file: {e}", file=sys.stderr)
        return 1

    # Create player and parse header
    player = VgmPlayer(output_rate=args.rate)
    data_start = player.parse_header(vgm_data)

    if not player.chips:
        print("No compatible chips found, exiting.", file=sys.stderr)
        return 1

    # Generate audio
    print(f"Generating audio at {args.rate}Hz...")
    samples = player.generate_all(vgm_data, data_start)

    # Write WAV
    print(f"Writing {len(samples)} samples to {args.output}...")
    write_wav(args.output, args.rate, samples)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
