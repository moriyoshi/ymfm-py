"""Tests for OPZ family chips.

OPZ (FM Operator Type-Z) chips include:
- YM2414 (OPZ) - 8 FM channels, 4 operators (TX81Z/DX11 synthesizer)
"""

import numpy as np
from conftest import ChipTestHelper

import ymfm

# =============================================================================
# YM2414 (OPZ) Tests
# =============================================================================


class TestYM2414:
    """Tests for YM2414 (OPZ) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2414, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2414(clock=3579545)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 2)
        assert samples.format == "i"

    def test_read_status(self):
        """Test status register read."""
        chip = ymfm.YM2414(clock=3579545)
        status = chip.read_status()
        assert isinstance(status, int)
        assert 0 <= status <= 255

    def test_write_operations(self):
        """Test write operations."""
        chip = ymfm.YM2414(clock=3579545)
        chip.write_address(0x20)
        chip.write_data(0x00)
        chip.write(0, 0x21)
        chip.write(1, 0xFF)

    def test_key_on_off(self):
        """Test key on/off."""
        chip = ymfm.YM2414(clock=3579545)
        chip.reset()
        # Key on register similar to OPM
        chip.write(0, 0x08)
        chip.write(1, 0x78)  # Channel 0, all operators
        samples = chip.generate(100)
        # Key off
        chip.write(0, 0x08)
        chip.write(1, 0x00)
        samples = chip.generate(100)
        assert samples is not None

    def test_channel_configuration(self):
        """Test channel configuration."""
        chip = ymfm.YM2414(clock=3579545)
        chip.reset()
        # Set connection and feedback for channel 0
        chip.write(0, 0x20)
        chip.write(1, 0xC7)  # RL=11, FB=0, CON=7
        samples = chip.generate(100)
        assert samples is not None

    def test_operator_configuration(self):
        """Test operator configuration."""
        chip = ymfm.YM2414(clock=3579545)
        chip.reset()
        # DT1/MUL for operator 0, channel 0
        chip.write(0, 0x40)
        chip.write(1, 0x01)
        # TL for operator 0, channel 0
        chip.write(0, 0x60)
        chip.write(1, 0x00)
        samples = chip.generate(100)
        assert samples is not None

    def test_lfo_configuration(self):
        """Test LFO configuration (similar to OPM)."""
        chip = ymfm.YM2414(clock=3579545)
        chip.reset()
        # LFO frequency
        chip.write(0, 0x18)
        chip.write(1, 0x00)
        # PMD/AMD
        chip.write(0, 0x19)
        chip.write(1, 0x00)
        samples = chip.generate(100)
        assert samples is not None

    def test_state_save_restore(self):
        """Test state save/restore."""
        chip = ymfm.YM2414(clock=3579545)
        chip.write(0, 0x28)
        chip.write(1, 0x00)
        chip.generate(10)

        state = chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0

        chip2 = ymfm.YM2414(clock=3579545)
        chip2.load_state(state)

    def test_reset(self):
        """Test chip reset."""
        chip = ymfm.YM2414(clock=3579545)
        chip.write(0, 0x20)
        chip.write(1, 0xFF)
        chip.reset()
        samples = chip.generate(100)
        assert samples is not None

    def test_interface_property(self):
        """Test interface property."""
        chip = ymfm.YM2414(clock=3579545)
        assert chip.interface is not None
        assert isinstance(chip.interface, ymfm.ChipInterface)

    def test_generate_zero_samples(self):
        """Test generating zero samples."""
        chip = ymfm.YM2414(clock=3579545)
        samples = chip.generate(0)
        assert samples.shape == (0, 2)

    def test_long_generation(self):
        """Test generating many samples."""
        chip = ymfm.YM2414(clock=3579545)
        samples = chip.generate(10000)
        assert samples.shape == (10000, 2)

    def test_multiple_resets(self):
        """Test multiple resets."""
        chip = ymfm.YM2414(clock=3579545)
        for _ in range(5):
            chip.reset()
            chip.generate(10)

    def test_generate_into(self):
        """Test generate_into method."""
        chip = ymfm.YM2414(clock=3579545)
        ChipTestHelper.test_generate_into(chip)

    def test_generate_into_matches_generate(self):
        """Test that generate_into produces same output as generate."""
        ChipTestHelper.test_generate_into_matches_generate(ymfm.YM2414, 3579545)

    def test_generate_into_zero_samples(self):
        """Test generate_into with zero-length buffer."""
        chip = ymfm.YM2414(clock=3579545)
        buffer = np.zeros((0, 2), dtype=np.int32)
        result = chip.generate_into(buffer)
        assert result == 0

    def test_generate_into_large_buffer(self):
        """Test generate_into with large buffer."""
        chip = ymfm.YM2414(clock=3579545)
        buffer = np.zeros((10000, 2), dtype=np.int32)
        result = chip.generate_into(buffer)
        assert result == 10000
