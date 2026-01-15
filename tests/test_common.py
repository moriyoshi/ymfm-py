"""Tests for common interfaces and shared functionality.

Tests cover:
- ChipInterface class
- MemoryInterface class
- SsgOverride class
- AccessClass enum
- generate_into error handling
"""

import numpy as np
import pytest

import ymfm

# =============================================================================
# ChipInterface Tests
# =============================================================================


class TestChipInterface:
    """Tests for ChipInterface class."""

    def test_create_interface(self):
        """Test creating a basic ChipInterface."""
        interface = ymfm.ChipInterface()
        assert interface is not None

    def test_chip_with_interface(self):
        """Test creating chip with custom interface."""
        interface = ymfm.ChipInterface()
        chip = ymfm.YM2612(clock=7670453, interface=interface)
        assert chip.interface is interface

    def test_multiple_chips_same_interface(self):
        """Test multiple chips can share an interface."""
        interface = ymfm.ChipInterface()
        chip1 = ymfm.YM2612(clock=7670453, interface=interface)
        chip2 = ymfm.YM3438(clock=7670453, interface=interface)
        assert chip1.interface is interface
        assert chip2.interface is interface


# =============================================================================
# MemoryInterface Tests
# =============================================================================


class TestMemoryInterface:
    """Tests for MemoryInterface class."""

    def test_create_interface(self):
        """Test creating a MemoryInterface."""
        interface = ymfm.MemoryInterface()
        assert interface is not None

    def test_set_memory(self):
        """Test setting memory."""
        interface = ymfm.MemoryInterface()
        test_data = b"\x00\x11\x22\x33\x44\x55"
        interface.set_memory(test_data)

    def test_get_memory(self):
        """Test getting memory."""
        interface = ymfm.MemoryInterface()
        test_data = b"\x00\x11\x22\x33\x44\x55"
        interface.set_memory(test_data)
        retrieved = interface.get_memory()
        assert retrieved == test_data

    def test_large_memory(self):
        """Test with large memory buffer."""
        interface = ymfm.MemoryInterface()
        test_data = bytes(1024 * 1024)  # 1MB
        interface.set_memory(test_data)
        retrieved = interface.get_memory()
        assert len(retrieved) == 1024 * 1024

    def test_chip_with_memory_interface(self):
        """Test chip with memory interface."""
        interface = ymfm.MemoryInterface()
        interface.set_memory(bytes(256 * 1024))
        chip = ymfm.YM2608(clock=8000000, interface=interface)
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# SsgOverride Tests
# =============================================================================


class TestSsgOverride:
    """Tests for SsgOverride class."""

    def test_create_ssg_override(self):
        """Test creating an SsgOverride instance."""
        override = ymfm.SsgOverride()
        assert override is not None

    def test_ssg_reset(self):
        """Test ssg_reset method."""
        override = ymfm.SsgOverride()
        override.ssg_reset()  # Should not raise

    def test_ssg_read(self):
        """Test ssg_read method."""
        override = ymfm.SsgOverride()
        data = override.ssg_read(0)
        assert isinstance(data, int)

    def test_ssg_write(self):
        """Test ssg_write method."""
        override = ymfm.SsgOverride()
        override.ssg_write(0, 0x00)  # Should not raise

    def test_ssg_prescale_changed(self):
        """Test ssg_prescale_changed method."""
        override = ymfm.SsgOverride()
        override.ssg_prescale_changed()  # Should not raise

    def test_custom_ssg_override(self):
        """Test custom SSG override subclass."""

        class CustomSsg(ymfm.SsgOverride):
            def __init__(self):
                super().__init__()
                self.registers = [0] * 16
                self.reset_called = False

            def ssg_reset(self):
                self.reset_called = True
                self.registers = [0] * 16

            def ssg_read(self, regnum):
                if 0 <= regnum < 16:
                    return self.registers[regnum]
                return 0

            def ssg_write(self, regnum, data):
                if 0 <= regnum < 16:
                    self.registers[regnum] = data

        custom = CustomSsg()
        assert not custom.reset_called

        custom.ssg_reset()
        assert custom.reset_called
        assert custom.registers == [0] * 16

        custom.ssg_write(0, 0x55)
        assert custom.ssg_read(0) == 0x55


# =============================================================================
# AccessClass Enum Tests
# =============================================================================


class TestAccessClass:
    """Tests for AccessClass enum."""

    def test_access_class_exists(self):
        """Test that AccessClass enum exists."""
        assert hasattr(ymfm, "AccessClass")


# =============================================================================
# Module-level Tests
# =============================================================================


class TestModuleInfo:
    """Tests for module-level information."""

    def test_version(self):
        """Test version string exists."""
        assert hasattr(ymfm, "__version__")
        assert isinstance(ymfm.__version__, str)

    def test_all_chips_exported(self):
        """Test all chips are exported."""
        expected_chips = [
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
            # Misc
            "YM2149",
            "YM2414",
            "YM3806",
            "YM3533",
        ]
        for chip in expected_chips:
            assert hasattr(ymfm, chip), f"Missing chip: {chip}"

    def test_interfaces_exported(self):
        """Test interfaces are exported."""
        assert hasattr(ymfm, "ChipInterface")
        assert hasattr(ymfm, "MemoryInterface")
        assert hasattr(ymfm, "SsgOverride")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_all_chips_can_generate_audio(self):
        """Test that all chips can generate audio."""
        chips = [
            (ymfm.YM3526, 3579545),
            (ymfm.Y8950, 3579545),
            (ymfm.YM3812, 3579545),
            (ymfm.YMF262, 14318180),
            (ymfm.YM2413, 3579545),
            (ymfm.YMF289B, 14318180),
            (ymfm.YMF278B, 33868800),
            (ymfm.YM2423, 3579545),
            (ymfm.YMF281, 3579545),
            (ymfm.DS1001, 3579545),
            (ymfm.YM2203, 4000000),
            (ymfm.YM2608, 8000000),
            (ymfm.YM2610, 8000000),
            (ymfm.YM2610B, 8000000),
            (ymfm.YM2612, 7670453),
            (ymfm.YM3438, 7670453),
            (ymfm.YMF276, 7670453),
            (ymfm.YMF288, 8000000),
            (ymfm.YM2151, 3579545),
            (ymfm.YM2164, 3579545),
            (ymfm.YM2149, 2000000),
            (ymfm.YM2414, 3579545),
            (ymfm.YM3806, 3579545),
            (ymfm.YM3533, 3579545),
        ]

        for chip_class, clock in chips:
            chip = chip_class(clock=clock)
            chip.reset()
            samples = chip.generate(100)
            assert samples is not None
            assert isinstance(samples, memoryview)
            assert samples.shape == (100, chip.outputs)

    def test_multiple_chips_simultaneous(self):
        """Test running multiple different chips simultaneously."""
        chip1 = ymfm.YM2612(clock=7670453)
        chip2 = ymfm.YM2151(clock=3579545)
        chip3 = ymfm.YM3812(clock=3579545)

        for _ in range(10):
            samples1 = chip1.generate(100)
            samples2 = chip2.generate(100)
            samples3 = chip3.generate(100)

            assert samples1.shape == (100, chip1.outputs)
            assert samples2.shape == (100, chip2.outputs)
            assert samples3.shape == (100, chip3.outputs)


# =============================================================================
# generate_into Error Handling Tests
# =============================================================================


class TestGenerateIntoErrors:
    """Tests for generate_into error handling."""

    @pytest.fixture
    def chip(self):
        """Provide a test chip instance."""
        return ymfm.YM2612(clock=7670453)

    def test_wrong_dtype_float32(self, chip):
        """Test error with wrong dtype (float32)."""
        buffer = np.zeros((10, chip.outputs), dtype=np.float32)
        with pytest.raises(ValueError, match="int32"):
            chip.generate_into(buffer)

    def test_wrong_dtype_float64(self, chip):
        """Test error with wrong dtype (float64)."""
        buffer = np.zeros((10, chip.outputs), dtype=np.float64)
        with pytest.raises(ValueError, match="int32"):
            chip.generate_into(buffer)

    def test_wrong_dtype_int16(self, chip):
        """Test error with wrong dtype (int16)."""
        buffer = np.zeros((10, chip.outputs), dtype=np.int16)
        with pytest.raises(ValueError, match="int32"):
            chip.generate_into(buffer)

    def test_wrong_dtype_int64(self, chip):
        """Test error with wrong dtype (int64)."""
        buffer = np.zeros((10, chip.outputs), dtype=np.int64)
        with pytest.raises(ValueError, match="int32"):
            chip.generate_into(buffer)

    def test_wrong_2d_shape(self, chip):
        """Test error with wrong 2D shape."""
        wrong_outputs = chip.outputs + 1
        buffer = np.zeros((10, wrong_outputs), dtype=np.int32)
        with pytest.raises(ValueError, match="outputs"):
            chip.generate_into(buffer)

    def test_non_divisible_1d_length(self, chip):
        """Test error when 1D length is not divisible by outputs."""
        if chip.outputs > 1:
            buffer = np.zeros(chip.outputs + 1, dtype=np.int32)
            with pytest.raises(ValueError, match="divisible"):
                chip.generate_into(buffer)

    def test_non_contiguous_fortran_order(self, chip):
        """Test error with non-contiguous (Fortran-order) array."""
        if chip.outputs == 1:
            pytest.skip("Fortran order is C-contiguous for single-output chips")
        buffer = np.zeros((10, chip.outputs), dtype=np.int32, order="F")
        with pytest.raises(ValueError, match="contiguous"):
            chip.generate_into(buffer)

    def test_non_contiguous_slice(self, chip):
        """Test error with non-contiguous slice."""
        full_buffer = np.zeros((20, chip.outputs), dtype=np.int32)
        sliced = full_buffer[::2]  # Every other row is non-contiguous
        with pytest.raises(ValueError, match="contiguous"):
            chip.generate_into(sliced)

    def test_3d_array(self, chip):
        """Test error with 3D array."""
        buffer = np.zeros((10, chip.outputs, 2), dtype=np.int32)
        with pytest.raises(ValueError, match="1D or 2D"):
            chip.generate_into(buffer)

    def test_readonly_buffer(self, chip):
        """Test error with read-only buffer."""
        buffer = np.zeros((10, chip.outputs), dtype=np.int32)
        buffer.flags.writeable = False
        with pytest.raises((ValueError, BufferError)):
            chip.generate_into(buffer)
