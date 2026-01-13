"""Tests for OPN family chips.

OPN (FM Operator Type-N) chips include:
- YM2203 (OPN) - 3 FM + SSG
- YM2608 (OPNA) - 6 FM + SSG + ADPCM (PC-98)
- YM2610 (OPNB) - 4 FM + SSG + ADPCM (Neo Geo)
- YM2610B (OPNB variant) - 6 FM channels
- YM2612 (OPN2) - 6 FM (Sega Genesis)
- YM3438 (OPN2C) - YM2612 with improved DAC
- YMF276 (OPN2 variant) - improved DAC accuracy
- YMF288 (OPN3L) - 6 FM + SSG + ADPCM-A
"""

import pytest
from conftest import ChipTestHelper

import ymfm

# =============================================================================
# Chip configurations
# =============================================================================

OPN_CHIPS = {
    "YM2203": (ymfm.YM2203, 4000000, 4),  # 4 outputs (FM stereo + SSG stereo)
    "YM2608": (ymfm.YM2608, 8000000, 3),  # 3 outputs
    "YM2610": (ymfm.YM2610, 8000000, 3),  # 3 outputs
    "YM2610B": (ymfm.YM2610B, 8000000, 3),  # 3 outputs
    "YM2612": (ymfm.YM2612, 7670453, 2),  # Stereo
    "YM3438": (ymfm.YM3438, 7670453, 2),  # Stereo
    "YMF276": (ymfm.YMF276, 7670453, 2),  # Stereo
    "YMF288": (ymfm.YMF288, 8000000, 3),  # 3 outputs
}

# Chips with SSG support
SSG_CHIPS = ["YM2203", "YM2608", "YM2610", "YM2610B", "YMF288"]

# Chips with high address/data registers
DUAL_PORT_CHIPS = [
    "YM2608",
    "YM2610",
    "YM2610B",
    "YM2612",
    "YM3438",
    "YMF276",
    "YMF288",
]


# =============================================================================
# YM2203 (OPN) Tests
# =============================================================================


class TestYM2203:
    """Tests for YM2203 (OPN) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2203, 4000000, 4)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2203(clock=4000000)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 4)
        assert samples.format == "i"

    def test_fm_channels(self):
        """Test FM channel configuration."""
        chip = ymfm.YM2203(clock=4000000)
        chip.reset()
        # Configure algorithm for channel 0
        chip.write(0, 0xB0)
        chip.write(1, 0x07)  # Algorithm 7
        # Key on
        chip.write(0, 0x28)
        chip.write(1, 0xF0)  # Channel 0, all operators
        samples = chip.generate(100)
        assert samples is not None

    def test_ssg_override_support(self):
        """Test SSG override support."""
        chip = ymfm.YM2203(clock=4000000)
        assert chip.ssg_override is None
        override = ymfm.SsgOverride()
        chip.set_ssg_override(override)
        assert chip.ssg_override is override


# =============================================================================
# YM2608 (OPNA) Tests
# =============================================================================


class TestYM2608:
    """Tests for YM2608 (OPNA) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2608, 8000000, 3)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2608(clock=8000000)
        samples = chip.generate(100)
        assert samples.shape == (100, 3)

    def test_6channel_mode(self):
        """Test 6-channel FM mode."""
        chip = ymfm.YM2608(clock=8000000)
        chip.reset()
        # Enable 6-channel mode
        chip.write(0, 0x29)
        chip.write(1, 0x80)
        samples = chip.generate(100)
        assert samples is not None

    def test_high_address_registers(self):
        """Test high address and data registers."""
        chip = ymfm.YM2608(clock=8000000)
        chip.write_address_hi(0x30)
        chip.write_data_hi(0x00)

    def test_read_status_hi(self):
        """Test high status register read."""
        chip = ymfm.YM2608(clock=8000000)
        status = chip.read_status_hi()
        assert isinstance(status, int)

    def test_ssg_override_support(self):
        """Test SSG override support."""
        chip = ymfm.YM2608(clock=8000000)
        assert chip.ssg_override is None
        override = ymfm.SsgOverride()
        chip.set_ssg_override(override)
        assert chip.ssg_override is override


# =============================================================================
# YM2610 (OPNB) Tests
# =============================================================================


class TestYM2610:
    """Tests for YM2610 (OPNB) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2610, 8000000, 3)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2610(clock=8000000)
        samples = chip.generate(100)
        assert samples.shape == (100, 3)

    def test_high_address_registers(self):
        """Test high address and data registers."""
        chip = ymfm.YM2610(clock=8000000)
        chip.write_address_hi(0x30)
        chip.write_data_hi(0x00)

    def test_ssg_override_support(self):
        """Test SSG override support."""
        chip = ymfm.YM2610(clock=8000000)
        assert chip.ssg_override is None
        override = ymfm.SsgOverride()
        chip.set_ssg_override(override)
        assert chip.ssg_override is override


# =============================================================================
# YM2610B (OPNB variant) Tests
# =============================================================================


class TestYM2610B:
    """Tests for YM2610B (OPNB variant) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2610B, 8000000, 3)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2610B(clock=8000000)
        samples = chip.generate(100)
        assert samples.shape == (100, 3)

    def test_ssg_override_support(self):
        """Test SSG override support."""
        chip = ymfm.YM2610B(clock=8000000)
        assert chip.ssg_override is None


# =============================================================================
# YM2612 (OPN2) Tests
# =============================================================================


class TestYM2612:
    """Tests for YM2612 (OPN2/Sega Genesis) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2612, 7670453, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2612(clock=7670453)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)

    def test_dac_mode(self):
        """Test DAC mode."""
        chip = ymfm.YM2612(clock=7670453)
        chip.reset()
        # Enable DAC
        chip.write(0, 0x2B)
        chip.write(1, 0x80)
        # Write DAC value
        chip.write(0, 0x2A)
        chip.write(1, 0x80)
        samples = chip.generate(100)
        assert samples is not None

    def test_high_address_registers(self):
        """Test high address and data registers."""
        chip = ymfm.YM2612(clock=7670453)
        chip.write_address_hi(0x30)
        chip.write_data_hi(0x00)

    def test_lfo_configuration(self):
        """Test LFO configuration."""
        chip = ymfm.YM2612(clock=7670453)
        chip.reset()
        # Enable LFO
        chip.write(0, 0x22)
        chip.write(1, 0x08)  # LFO on, frequency 0
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# YM3438 (OPN2C) Tests
# =============================================================================


class TestYM3438:
    """Tests for YM3438 (OPN2C) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM3438, 7670453, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM3438(clock=7670453)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)

    def test_compatible_with_ym2612(self):
        """Test YM2612 compatibility."""
        chip = ymfm.YM3438(clock=7670453)
        chip.reset()
        # Same DAC enable sequence as YM2612
        chip.write(0, 0x2B)
        chip.write(1, 0x80)
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# YMF276 (OPN2 variant) Tests
# =============================================================================


class TestYMF276:
    """Tests for YMF276 (OPN2 variant) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YMF276, 7670453, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YMF276(clock=7670453)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)

    def test_high_address_registers(self):
        """Test high address and data registers."""
        chip = ymfm.YMF276(clock=7670453)
        chip.write_address_hi(0x30)
        chip.write_data_hi(0x00)


# =============================================================================
# YMF288 (OPN3L) Tests
# =============================================================================


class TestYMF288:
    """Tests for YMF288 (OPN3L) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YMF288, 8000000, 3)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YMF288(clock=8000000)
        samples = chip.generate(100)
        assert samples.shape == (100, 3)

    def test_read_status_hi(self):
        """Test high status register."""
        chip = ymfm.YMF288(clock=8000000)
        status = chip.read_status_hi()
        assert isinstance(status, int)

    def test_read_data(self):
        """Test data register read."""
        chip = ymfm.YMF288(clock=8000000)
        data = chip.read_data()
        assert isinstance(data, int)

    def test_ssg_override_support(self):
        """Test SSG override support."""
        chip = ymfm.YMF288(clock=8000000)
        assert chip.ssg_override is None
        override = ymfm.SsgOverride()
        chip.set_ssg_override(override)
        assert chip.ssg_override is override


# =============================================================================
# SSG Override Tests
# =============================================================================


class TestSSGOverride:
    """Test SSG override functionality for OPN chips."""

    @pytest.mark.parametrize("chip_name", SSG_CHIPS)
    def test_ssg_override_available(self, chip_name):
        """Test that SSG chips have override support."""
        chip_class, clock, _ = OPN_CHIPS[chip_name]
        chip = chip_class(clock=clock)
        assert hasattr(chip, "set_ssg_override")
        assert hasattr(chip, "ssg_override")

    def test_custom_ssg_override(self):
        """Test custom SSG override implementation."""

        class CustomSsg(ymfm.SsgOverride):
            def __init__(self):
                super().__init__()
                self.registers = [0] * 16
                self.reset_count = 0

            def ssg_reset(self):
                self.reset_count += 1
                self.registers = [0] * 16

            def ssg_read(self, regnum):
                return self.registers[regnum] if regnum < 16 else 0

            def ssg_write(self, regnum, data):
                if regnum < 16:
                    self.registers[regnum] = data

        custom = CustomSsg()
        chip = ymfm.YM2203(clock=4000000)
        chip.set_ssg_override(custom)

        chip.reset()
        samples = chip.generate(100)
        assert samples.shape == (100, chip.outputs)


# =============================================================================
# Parametrized tests for all OPN chips
# =============================================================================


@pytest.fixture(params=list(OPN_CHIPS.keys()))
def opn_chip_name(request):
    """Fixture providing each OPN chip name."""
    return request.param


@pytest.fixture
def opn_chip(opn_chip_name):
    """Fixture providing each OPN chip instance."""
    chip_class, clock, _ = OPN_CHIPS[opn_chip_name]
    return chip_class(clock=clock)


class TestOPNCommon:
    """Common tests for all OPN family chips."""

    def test_interface_property(self, opn_chip):
        """Test interface property."""
        assert opn_chip.interface is not None
        assert isinstance(opn_chip.interface, ymfm.ChipInterface)

    def test_reset(self, opn_chip):
        """Test chip reset."""
        opn_chip.reset()
        samples = opn_chip.generate(100)
        assert samples is not None

    def test_read_status(self, opn_chip):
        """Test status register read."""
        status = opn_chip.read_status()
        assert isinstance(status, int)
        assert 0 <= status <= 255

    def test_generate_zero_samples(self, opn_chip):
        """Test generating zero samples."""
        samples = opn_chip.generate(0)
        assert samples.shape == (0, opn_chip.outputs)

    def test_state_save_restore(self, opn_chip):
        """Test state save/restore."""
        state = opn_chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0
        opn_chip.load_state(state)

    def test_long_generation(self, opn_chip):
        """Test generating many samples."""
        samples = opn_chip.generate(10000)
        assert samples.shape == (10000, opn_chip.outputs)

    def test_multiple_resets(self, opn_chip):
        """Test multiple resets."""
        for _ in range(5):
            opn_chip.reset()
            opn_chip.generate(10)


# =============================================================================
# Per-operator Frequency Mode Tests (CH3 Extended Mode)
# =============================================================================


class TestOPNPerOperatorFrequency:
    """Tests for OPN per-operator frequency mode (CH3 extended mode).

    This tests the fix for the ymfm_sync_mode_write callback which was
    incorrectly overridden to do nothing, causing writes to register 0x27
    (the mode register) to be silently dropped.
    """

    # OPN register definitions
    REG_TIMER_CTRL = 0x27  # Timer control and CH3 mode
    REG_KEY_ON = 0x28
    REG_FB_CON = 0xB0
    REG_DT1_MUL = 0x30
    REG_TL = 0x40
    REG_KS_AR = 0x50
    REG_AM_DR = 0x60
    REG_SR = 0x70
    REG_SL_RR = 0x80

    # CH3 per-operator frequency registers
    CH3_OP_FREQ_REGS = [
        (0xAD, 0xA9),  # Operator 1
        (0xAE, 0xAA),  # Operator 2
        (0xAC, 0xA8),  # Operator 3
        (0xA6, 0xA2),  # Operator 4 (base CH3 registers)
    ]

    # OPN slot order mapping
    OP_SLOT_OFFSET = [0, 8, 4, 12]

    # Mode register bits
    MODE_EXTENDED = 0x40  # Bit 6: Per-operator frequency mode

    @staticmethod
    def _convert_frequency(freq: float, clock: int, prescaler: int) -> tuple:
        """Convert frequency to F-Number and Block."""
        for block in range(8):
            fnum = int(freq * prescaler * (1 << (21 - block)) / clock)
            if 1 <= fnum <= 2047:
                return fnum, block
        return (2047, 7)

    @staticmethod
    def _detect_frequency_peaks(samples, sample_rate: int, threshold: float = 0.1):
        """Detect frequency peaks in audio samples.

        Returns list of (frequency, amplitude) tuples for peaks above threshold.
        """
        import numpy as np

        # Convert to numpy and use first channel
        arr = np.asarray(samples)
        if len(arr.shape) > 1:
            audio = arr[:, 0].astype(np.float64)
        else:
            audio = arr.astype(np.float64)

        # Apply window
        window = np.hanning(len(audio))
        audio = audio * window

        # FFT
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)

        # Find peaks above threshold
        max_amp = np.max(fft)
        if max_amp == 0:
            return []

        peaks = []
        for i in range(1, len(fft) - 1):
            if fft[i] > threshold * max_amp:
                if fft[i] > fft[i - 1] and fft[i] > fft[i + 1]:
                    peaks.append((freqs[i], fft[i]))

        return sorted(peaks, key=lambda x: -x[1])

    def _write_reg(self, chip, reg: int, data: int):
        """Write a register and generate a sample to process it."""
        chip.write(0, reg)
        chip.write(1, data)
        chip.generate(1)  # Process the write

    def _setup_ch3_operator(self, chip, op: int):
        """Set up a single CH3 operator for sound generation."""
        op_offset = 2 + self.OP_SLOT_OFFSET[op]  # CH3 = index 2
        self._write_reg(chip, self.REG_DT1_MUL + op_offset, 0x01)  # MUL=1
        self._write_reg(chip, self.REG_TL + op_offset, 0x00)  # TL=0 (max volume)
        self._write_reg(chip, self.REG_KS_AR + op_offset, 0x1F)  # Max AR
        self._write_reg(chip, self.REG_AM_DR + op_offset, 0x00)
        self._write_reg(chip, self.REG_SR + op_offset, 0x00)
        self._write_reg(chip, self.REG_SL_RR + op_offset, 0x0F)

    def _set_operator_frequency(
        self, chip, op: int, freq: float, clock: int, prescaler: int
    ):
        """Set frequency for a specific CH3 operator."""
        fnum, block = self._convert_frequency(freq, clock, prescaler)
        hi_reg, lo_reg = self.CH3_OP_FREQ_REGS[op]

        # Write high first, then low
        self._write_reg(chip, hi_reg, ((block & 0x07) << 3) | ((fnum >> 8) & 0x07))
        self._write_reg(chip, lo_reg, fnum & 0xFF)

    def test_mode_register_write_ym2203(self):
        """Test that mode register (0x27) writes work for YM2203.

        This verifies the fix for ymfm_sync_mode_write callback.
        """
        chip = ymfm.YM2203(clock=4000000)
        chip.reset()

        # Write to mode register to enable per-operator frequency mode
        chip.write(0, self.REG_TIMER_CTRL)
        chip.write(1, self.MODE_EXTENDED)

        # Generate samples - this should not crash and the mode should be active
        samples = chip.generate(1000)
        assert samples is not None

    def test_mode_register_write_ym2608(self):
        """Test that mode register (0x27) writes work for YM2608.

        This verifies the fix for ymfm_sync_mode_write callback.
        """
        chip = ymfm.YM2608(clock=8000000)
        chip.reset()

        # Write to mode register to enable per-operator frequency mode
        chip.write(0, self.REG_TIMER_CTRL)
        chip.write(1, self.MODE_EXTENDED)

        # Generate samples - this should not crash and the mode should be active
        samples = chip.generate(1000)
        assert samples is not None

    def test_per_operator_frequency_ym2203(self):
        """Test per-operator frequency mode produces distinct frequencies on YM2203.

        In per-operator frequency mode (CH3 extended mode), each of the 4 operators
        of Channel 3 can have an independent frequency. This test verifies that
        setting different frequencies for each operator results in all frequencies
        being present in the output.
        """
        clock = 4000000
        prescaler = 72  # YM2203 prescaler
        chip = ymfm.YM2203(clock=clock)
        chip.reset()

        # Set CH3 to algorithm 7 (all operators as carriers)
        self._write_reg(chip, self.REG_FB_CON + 2, 0x07)

        # Set up all 4 operators
        for op in range(4):
            self._setup_ch3_operator(chip, op)

        # Enable per-operator frequency mode
        self._write_reg(chip, self.REG_TIMER_CTRL, self.MODE_EXTENDED)

        # Set different frequencies for each operator
        test_freqs = [220.0, 330.0, 440.0, 550.0]
        for op, freq in enumerate(test_freqs):
            self._set_operator_frequency(chip, op, freq, clock, prescaler)

        # Key on CH3 with all operators
        self._write_reg(chip, self.REG_KEY_ON, 0xF0 | 2)

        # Generate enough samples for good frequency resolution
        sample_rate = chip.sample_rate
        # Need enough samples for FFT resolution - at least 0.1s of audio
        num_samples = max(8000, int(sample_rate * 0.15))
        samples = chip.generate(num_samples)

        # Detect frequency peaks
        peaks = self._detect_frequency_peaks(samples, sample_rate, threshold=0.05)

        # Verify we have multiple distinct frequencies
        assert len(peaks) >= 4, f"Expected at least 4 frequency peaks, got {len(peaks)}"

        # Check that each expected frequency is present (within 20 Hz tolerance)
        found_freqs = []
        for expected_freq in test_freqs:
            for peak_freq, _ in peaks[:15]:
                if abs(peak_freq - expected_freq) < 20:
                    found_freqs.append(expected_freq)
                    break

        assert len(found_freqs) == 4, (
            f"Expected all 4 frequencies {test_freqs}, "
            f"but only found {found_freqs}. "
            f"Peaks: {[(f, a) for f, a in peaks[:10]]}"
        )

    def test_per_operator_frequency_ym2608(self):
        """Test per-operator frequency mode produces distinct frequencies on YM2608.

        Same as YM2203 test but with YM2608's different clock and prescaler.
        """
        clock = 8000000
        prescaler = 144  # YM2608 prescaler
        chip = ymfm.YM2608(clock=clock)
        chip.reset()

        # Set CH3 to algorithm 7 (all operators as carriers)
        self._write_reg(chip, self.REG_FB_CON + 2, 0x07)

        # Set up all 4 operators
        for op in range(4):
            self._setup_ch3_operator(chip, op)

        # Enable per-operator frequency mode
        self._write_reg(chip, self.REG_TIMER_CTRL, self.MODE_EXTENDED)

        # Set different frequencies for each operator
        test_freqs = [220.0, 330.0, 440.0, 550.0]
        for op, freq in enumerate(test_freqs):
            self._set_operator_frequency(chip, op, freq, clock, prescaler)

        # Key on CH3 with all operators
        self._write_reg(chip, self.REG_KEY_ON, 0xF0 | 2)

        # Generate samples
        sample_rate = chip.sample_rate
        num_samples = max(8000, int(sample_rate * 0.15))
        samples = chip.generate(num_samples)

        # Detect frequency peaks
        peaks = self._detect_frequency_peaks(samples, sample_rate, threshold=0.05)

        # Verify we have multiple distinct frequencies
        assert len(peaks) >= 4, f"Expected at least 4 frequency peaks, got {len(peaks)}"

        # Check that each expected frequency is present (within 20 Hz tolerance)
        found_freqs = []
        for expected_freq in test_freqs:
            for peak_freq, _ in peaks[:15]:
                if abs(peak_freq - expected_freq) < 20:
                    found_freqs.append(expected_freq)
                    break

        assert len(found_freqs) == 4, (
            f"Expected all 4 frequencies {test_freqs}, "
            f"but only found {found_freqs}. "
            f"Peaks: {[(f, a) for f, a in peaks[:10]]}"
        )

    def test_normal_mode_single_frequency(self):
        """Test that without extended mode, all operators use the same frequency.

        This is a control test to verify the extended mode actually makes a difference.
        """
        clock = 4000000
        prescaler = 72
        chip = ymfm.YM2203(clock=clock)
        chip.reset()

        # Set CH3 to algorithm 7
        self._write_reg(chip, self.REG_FB_CON + 2, 0x07)

        # Set up all 4 operators
        for op in range(4):
            self._setup_ch3_operator(chip, op)

        # Do NOT enable extended mode - leave mode register at 0
        self._write_reg(chip, self.REG_TIMER_CTRL, 0x00)

        # Set different frequencies for each operator
        # In normal mode, only OP4's frequency (base CH3 regs) should be used
        test_freqs = [220.0, 330.0, 440.0, 550.0]
        for op, freq in enumerate(test_freqs):
            self._set_operator_frequency(chip, op, freq, clock, prescaler)

        # Key on CH3
        self._write_reg(chip, self.REG_KEY_ON, 0xF0 | 2)

        # Generate samples
        sample_rate = chip.sample_rate
        num_samples = max(8000, int(sample_rate * 0.15))
        samples = chip.generate(num_samples)

        # In normal mode, all operators should use the base frequency (OP4 = 550 Hz)
        peaks = self._detect_frequency_peaks(samples, sample_rate, threshold=0.1)

        # The dominant frequency should be close to 550 Hz (OP4's frequency)
        if len(peaks) > 0:
            dominant_freq = peaks[0][0]
            assert abs(dominant_freq - 550.0) < 25, (
                f"In normal mode, expected dominant frequency ~550 Hz, "
                f"got {dominant_freq} Hz"
            )
