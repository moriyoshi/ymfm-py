"""Tests for OPQ family chips.

OPQ (FM Operator Type-Q) chips include:
- YM3806 (OPQ) - 8 FM channels, 2 operators (arcade)
- YM3533 (OPQ variant) - 8 FM channels, 2 operators (arcade)
"""

import numpy as np
import pytest
from conftest import ChipTestHelper

import ymfm

# =============================================================================
# Chip configurations
# =============================================================================

OPQ_CHIPS = {
    "YM3806": (ymfm.YM3806, 3579545, 2),
    "YM3533": (ymfm.YM3533, 3579545, 2),
}


# =============================================================================
# YM3806 (OPQ) Tests
# =============================================================================


class TestYM3806:
    """Tests for YM3806 (OPQ) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM3806, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM3806(clock=3579545)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 2)
        assert samples.format == "i"

    def test_read_status(self):
        """Test status register read."""
        chip = ymfm.YM3806(clock=3579545)
        status = chip.read_status()
        assert isinstance(status, int)
        assert 0 <= status <= 255

    def test_write_operations(self):
        """Test write operations."""
        chip = ymfm.YM3806(clock=3579545)
        chip.write_address(0x00)
        chip.write_data(0x00)
        chip.write(0, 0x01)
        chip.write(1, 0xFF)

    def test_channel_enable(self):
        """Test channel enable."""
        chip = ymfm.YM3806(clock=3579545)
        chip.reset()
        # Enable outputs for channel 0
        chip.write(0, 0x10)
        chip.write(1, 0xC0)
        samples = chip.generate(100)
        assert samples is not None

    def test_state_save_restore(self):
        """Test state save/restore."""
        chip = ymfm.YM3806(clock=3579545)
        chip.generate(10)

        state = chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0

        chip2 = ymfm.YM3806(clock=3579545)
        chip2.load_state(state)

    def test_interface_property(self):
        """Test interface property."""
        chip = ymfm.YM3806(clock=3579545)
        assert chip.interface is not None
        assert isinstance(chip.interface, ymfm.ChipInterface)


# =============================================================================
# YM3533 (OPQ variant) Tests
# =============================================================================


class TestYM3533:
    """Tests for YM3533 (OPQ variant) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM3533, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM3533(clock=3579545)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 2)
        assert samples.format == "i"

    def test_read_status(self):
        """Test status register read."""
        chip = ymfm.YM3533(clock=3579545)
        status = chip.read_status()
        assert isinstance(status, int)
        assert 0 <= status <= 255

    def test_write_operations(self):
        """Test write operations."""
        chip = ymfm.YM3533(clock=3579545)
        chip.write_address(0x00)
        chip.write_data(0x00)
        chip.write(0, 0x01)
        chip.write(1, 0xFF)

    def test_state_save_restore(self):
        """Test state save/restore."""
        chip = ymfm.YM3533(clock=3579545)
        chip.generate(10)

        state = chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0

        chip2 = ymfm.YM3533(clock=3579545)
        chip2.load_state(state)

    def test_interface_property(self):
        """Test interface property."""
        chip = ymfm.YM3533(clock=3579545)
        assert chip.interface is not None
        assert isinstance(chip.interface, ymfm.ChipInterface)


# =============================================================================
# Parametrized tests for all OPQ chips
# =============================================================================


@pytest.fixture(params=list(OPQ_CHIPS.keys()))
def opq_chip_name(request):
    """Fixture providing each OPQ chip name."""
    return request.param


@pytest.fixture
def opq_chip(opq_chip_name):
    """Fixture providing each OPQ chip instance."""
    chip_class, clock, _ = OPQ_CHIPS[opq_chip_name]
    return chip_class(clock=clock)


class TestOPQCommon:
    """Common tests for all OPQ family chips."""

    def test_reset(self, opq_chip):
        """Test chip reset."""
        opq_chip.reset()
        samples = opq_chip.generate(100)
        assert samples is not None

    def test_generate_zero_samples(self, opq_chip):
        """Test generating zero samples."""
        samples = opq_chip.generate(0)
        assert samples.shape == (0, opq_chip.outputs)

    def test_long_generation(self, opq_chip):
        """Test generating many samples."""
        samples = opq_chip.generate(10000)
        assert samples.shape == (10000, opq_chip.outputs)

    def test_multiple_resets(self, opq_chip):
        """Test multiple resets."""
        for _ in range(5):
            opq_chip.reset()
            opq_chip.generate(10)

    def test_generate_into(self, opq_chip):
        """Test generate_into method."""
        ChipTestHelper.test_generate_into(opq_chip)

    def test_generate_into_matches_generate(self, opq_chip_name):
        """Test that generate_into produces same output as generate."""
        chip_class, clock, _ = OPQ_CHIPS[opq_chip_name]
        ChipTestHelper.test_generate_into_matches_generate(chip_class, clock)

    def test_generate_into_zero_samples(self, opq_chip):
        """Test generate_into with zero-length buffer."""
        buffer = np.zeros((0, opq_chip.outputs), dtype=np.int32)
        result = opq_chip.generate_into(buffer)
        assert result == 0

    def test_generate_into_large_buffer(self, opq_chip):
        """Test generate_into with large buffer."""
        buffer = np.zeros((10000, opq_chip.outputs), dtype=np.int32)
        result = opq_chip.generate_into(buffer)
        assert result == 10000
