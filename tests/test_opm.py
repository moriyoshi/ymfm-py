"""Tests for OPM family chips.

OPM (FM Operator Type-M) chips include:
- YM2151 (OPM) - 8 FM channels, stereo (Sharp X68000, arcade)
- YM2164 (OPP) - YM2151 variant with half-speed Timer B
"""

import numpy as np
import pytest
from conftest import ChipTestHelper

import ymfm

# =============================================================================
# Chip configurations
# =============================================================================

OPM_CHIPS = {
    "YM2151": (ymfm.YM2151, 3579545, 2),
    "YM2164": (ymfm.YM2164, 3579545, 2),
}


# =============================================================================
# YM2151 (OPM) Tests
# =============================================================================


class TestYM2151:
    """Tests for YM2151 (OPM) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2151, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2151(clock=3579545)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 2)
        assert samples.format == "i"

    def test_read_status(self):
        """Test status register read."""
        chip = ymfm.YM2151(clock=3579545)
        status = chip.read_status()
        assert isinstance(status, int)
        assert 0 <= status <= 255

    def test_lfo_configuration(self):
        """Test LFO configuration."""
        chip = ymfm.YM2151(clock=3579545)
        chip.reset()
        # LFO frequency
        chip.write(0, 0x18)
        chip.write(1, 0x00)
        # PMD/AMD
        chip.write(0, 0x19)
        chip.write(1, 0x00)
        # CT/W (control/waveform)
        chip.write(0, 0x1B)
        chip.write(1, 0x00)
        samples = chip.generate(100)
        assert samples is not None

    def test_key_on_off(self):
        """Test key on/off."""
        chip = ymfm.YM2151(clock=3579545)
        chip.reset()
        # Key on channel 0, all operators
        chip.write(0, 0x08)
        chip.write(1, 0x78)  # SN=0111, CH=0
        samples = chip.generate(100)
        # Key off
        chip.write(0, 0x08)
        chip.write(1, 0x00)
        samples = chip.generate(100)
        assert samples is not None

    def test_channel_configuration(self):
        """Test channel configuration."""
        chip = ymfm.YM2151(clock=3579545)
        chip.reset()
        # Set connection and feedback for channel 0
        chip.write(0, 0x20)
        chip.write(1, 0xC7)  # RL=11, FB=0, CON=7
        samples = chip.generate(100)
        assert samples is not None

    def test_operator_configuration(self):
        """Test operator configuration."""
        chip = ymfm.YM2151(clock=3579545)
        chip.reset()
        # DT1/MUL for operator 0, channel 0
        chip.write(0, 0x40)
        chip.write(1, 0x01)  # MUL=1
        # TL for operator 0, channel 0
        chip.write(0, 0x60)
        chip.write(1, 0x00)  # Full volume
        samples = chip.generate(100)
        assert samples is not None

    def test_state_save_restore(self):
        """Test state save/restore."""
        chip = ymfm.YM2151(clock=3579545)
        chip.reset()
        chip.write(0, 0x20)
        chip.write(1, 0xC7)
        chip.generate(50)

        state = chip.save_state()
        samples1 = chip.generate(100)

        chip.load_state(state)
        samples2 = chip.generate(100)

        np.testing.assert_array_equal(samples1, samples2)


# =============================================================================
# YM2164 (OPP) Tests
# =============================================================================


class TestYM2164:
    """Tests for YM2164 (OPP) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2164, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2164(clock=3579545)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 2)
        assert samples.format == "i"

    def test_ym2151_compatible(self):
        """Test YM2151 register compatibility."""
        chip = ymfm.YM2164(clock=3579545)
        chip.reset()
        # Same LFO configuration as YM2151
        chip.write(0, 0x18)
        chip.write(1, 0x00)
        chip.write(0, 0x19)
        chip.write(1, 0x00)
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# Parametrized tests for all OPM chips
# =============================================================================


@pytest.fixture(params=list(OPM_CHIPS.keys()))
def opm_chip_name(request):
    """Fixture providing each OPM chip name."""
    return request.param


@pytest.fixture
def opm_chip(opm_chip_name):
    """Fixture providing each OPM chip instance."""
    chip_class, clock, _ = OPM_CHIPS[opm_chip_name]
    return chip_class(clock=clock)


class TestOPMCommon:
    """Common tests for all OPM family chips."""

    def test_interface_property(self, opm_chip):
        """Test interface property."""
        assert opm_chip.interface is not None
        assert isinstance(opm_chip.interface, ymfm.ChipInterface)

    def test_reset(self, opm_chip):
        """Test chip reset."""
        opm_chip.reset()
        samples = opm_chip.generate(100)
        assert samples is not None

    def test_read_status(self, opm_chip):
        """Test status register read."""
        status = opm_chip.read_status()
        assert isinstance(status, int)

    def test_generate_zero_samples(self, opm_chip):
        """Test generating zero samples."""
        samples = opm_chip.generate(0)
        assert samples.shape == (0, opm_chip.outputs)

    def test_state_save_restore(self, opm_chip):
        """Test state save/restore."""
        state = opm_chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0
        opm_chip.load_state(state)

    def test_long_generation(self, opm_chip):
        """Test generating many samples."""
        samples = opm_chip.generate(10000)
        assert samples.shape == (10000, opm_chip.outputs)

    def test_multiple_resets(self, opm_chip):
        """Test multiple resets."""
        for _ in range(5):
            opm_chip.reset()
            opm_chip.generate(10)

    def test_generate_into(self, opm_chip):
        """Test generate_into method."""
        ChipTestHelper.test_generate_into(opm_chip)

    def test_generate_into_matches_generate(self, opm_chip_name):
        """Test that generate_into produces same output as generate."""
        chip_class, clock, _ = OPM_CHIPS[opm_chip_name]
        ChipTestHelper.test_generate_into_matches_generate(chip_class, clock)

    def test_generate_into_zero_samples(self, opm_chip):
        """Test generate_into with zero-length buffer."""
        buffer = np.zeros((0, opm_chip.outputs), dtype=np.int32)
        result = opm_chip.generate_into(buffer)
        assert result == 0

    def test_generate_into_large_buffer(self, opm_chip):
        """Test generate_into with large buffer."""
        buffer = np.zeros((10000, opm_chip.outputs), dtype=np.int32)
        result = opm_chip.generate_into(buffer)
        assert result == 10000
