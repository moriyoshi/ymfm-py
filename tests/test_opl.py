"""Tests for OPL family chips.

OPL (FM Operator Type-L) chips include:
- YM3526 (OPL) - 9 FM channels, mono
- Y8950 (MSX-Audio) - 9 FM + ADPCM
- YM3812 (OPL2) - 9 FM, 4 waveforms (Sound Blaster)
- YMF262 (OPL3) - 18 FM, 4-channel output
- YMF289B (OPL3L) - OPL3 variant, stereo
- YMF278B (OPL4) - OPL3 + wavetable PCM
- YM2413 (OPLL) - 9 FM with preset instruments
- YM2423 (OPLL variant)
- YMF281 (OPLLP)
- DS1001 (VRC7) - Konami OPLL for NES
"""

import pytest
from conftest import ChipTestHelper

import ymfm

# =============================================================================
# Chip configurations
# =============================================================================

OPL_CHIPS = {
    "YM3526": (ymfm.YM3526, 3579545, 1),
    "Y8950": (ymfm.Y8950, 3579545, 1),
    "YM3812": (ymfm.YM3812, 3579545, 1),
    "YMF262": (ymfm.YMF262, 14318180, 4),
    "YM2413": (ymfm.YM2413, 3579545, 2),
}

OPL3_CHIPS = {
    "YMF289B": (ymfm.YMF289B, 14318180, 2),
    "YMF278B": (ymfm.YMF278B, 33868800, 6),
}

OPLL_CHIPS = {
    "YM2423": (ymfm.YM2423, 3579545, 2),
    "YMF281": (ymfm.YMF281, 3579545, 2),
    "DS1001": (ymfm.DS1001, 3579545, 2),
}

ALL_OPL_CHIPS = {**OPL_CHIPS, **OPL3_CHIPS, **OPLL_CHIPS}


# =============================================================================
# YM3526 (OPL) Tests
# =============================================================================


class TestYM3526:
    """Tests for YM3526 (OPL) chip."""

    def test_create(self):
        """Test chip creation."""
        chip = ChipTestHelper.test_basic_instantiation(ymfm.YM3526, 3579545, 1)
        assert chip.outputs == 1  # Mono

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM3526(clock=3579545)
        ChipTestHelper.test_generate_samples(chip)

    def test_read_status(self):
        """Test status register read."""
        chip = ymfm.YM3526(clock=3579545)
        status = chip.read_status()
        assert isinstance(status, int)
        assert 0 <= status <= 255

    def test_register_write(self):
        """Test register write operations."""
        chip = ymfm.YM3526(clock=3579545)
        ChipTestHelper.test_register_operations(chip)

    def test_state_save_restore(self):
        """Test state save/restore."""
        chip = ymfm.YM3526(clock=3579545)
        state = chip.save_state()
        assert isinstance(state, bytes)
        chip.load_state(state)


# =============================================================================
# Y8950 (MSX-Audio) Tests
# =============================================================================


class TestY8950:
    """Tests for Y8950 (MSX-Audio) chip."""

    def test_create(self):
        """Test chip creation."""
        chip = ChipTestHelper.test_basic_instantiation(ymfm.Y8950, 3579545, 1)
        assert chip.outputs == 1

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.Y8950(clock=3579545)
        ChipTestHelper.test_generate_samples(chip)

    def test_with_memory_interface(self):
        """Test with memory interface for ADPCM."""
        interface = ymfm.MemoryInterface()
        interface.set_memory(bytes(256 * 1024))  # 256KB ADPCM RAM
        chip = ymfm.Y8950(clock=3579545, interface=interface)
        samples = chip.generate(100)
        assert samples.shape == (100, 1)


# =============================================================================
# YM3812 (OPL2) Tests
# =============================================================================


class TestYM3812:
    """Tests for YM3812 (OPL2) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM3812, 3579545, 1)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM3812(clock=3579545)
        ChipTestHelper.test_generate_samples(chip)

    def test_waveform_select(self):
        """Test waveform selection (OPL2 feature)."""
        chip = ymfm.YM3812(clock=3579545)
        chip.reset()
        # Enable waveform selection
        chip.write(0, 0x01)
        chip.write(1, 0x20)
        # Set waveform for operator 0
        chip.write(0, 0xE0)
        chip.write(1, 0x01)  # Half-sine
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# YMF262 (OPL3) Tests
# =============================================================================


class TestYMF262:
    """Tests for YMF262 (OPL3) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YMF262, 14318180, 4)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YMF262(clock=14318180)
        samples = chip.generate(100)
        assert samples.shape == (100, 4)

    def test_opl3_mode_enable(self):
        """Test OPL3 mode enable."""
        chip = ymfm.YMF262(clock=14318180)
        chip.reset()
        # Enable OPL3 mode
        chip.write(2, 0x05)  # Register 0x105
        chip.write(3, 0x01)  # NEW=1
        samples = chip.generate(100)
        assert samples is not None

    def test_4op_mode(self):
        """Test 4-operator mode configuration."""
        chip = ymfm.YMF262(clock=14318180)
        chip.reset()
        # Enable OPL3 mode
        chip.write(2, 0x05)
        chip.write(3, 0x01)
        # Enable 4-op mode for channel pair
        chip.write(2, 0x04)  # Register 0x104
        chip.write(3, 0x01)  # Enable 4-op for channels 0-1
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# YM2413 (OPLL) Tests
# =============================================================================


class TestYM2413:
    """Tests for YM2413 (OPLL) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2413, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2413(clock=3579545)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)

    def test_preset_instrument(self):
        """Test preset instrument selection."""
        chip = ymfm.YM2413(clock=3579545)
        chip.reset()
        # Select preset instrument 1 for channel 0
        chip.write(0, 0x30)
        chip.write(1, 0x10)  # Instrument 1, volume 0
        samples = chip.generate(100)
        assert samples is not None

    def test_custom_instrument(self):
        """Test custom instrument programming."""
        chip = ymfm.YM2413(clock=3579545)
        chip.reset()
        # Program custom instrument in registers 0x00-0x07
        chip.write(0, 0x00)
        chip.write(1, 0x21)  # Modulator settings
        chip.write(0, 0x01)
        chip.write(1, 0x21)  # Carrier settings
        samples = chip.generate(100)
        assert samples is not None


# =============================================================================
# YMF289B (OPL3L) Tests
# =============================================================================


class TestYMF289B:
    """Tests for YMF289B (OPL3L) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YMF289B, 14318180, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YMF289B(clock=14318180)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)

    def test_write_address_hi(self):
        """Test high address register."""
        chip = ymfm.YMF289B(clock=14318180)
        chip.write_address_hi(0x01)

    def test_read_data(self):
        """Test data register read."""
        chip = ymfm.YMF289B(clock=14318180)
        data = chip.read_data()
        assert isinstance(data, int)


# =============================================================================
# YMF278B (OPL4) Tests
# =============================================================================


class TestYMF278B:
    """Tests for YMF278B (OPL4) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YMF278B, 33868800, 6)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YMF278B(clock=33868800)
        samples = chip.generate(100)
        assert samples.shape == (100, 6)

    def test_pcm_registers(self):
        """Test PCM address and data registers."""
        chip = ymfm.YMF278B(clock=33868800)
        chip.write_address_pcm(0x00)
        chip.write_data_pcm(0x00)
        data = chip.read_data_pcm()
        assert isinstance(data, int)

    def test_with_memory_interface(self):
        """Test with memory interface for PCM data."""
        interface = ymfm.MemoryInterface()
        interface.set_memory(bytes(4 * 1024 * 1024))  # 4MB PCM ROM
        chip = ymfm.YMF278B(clock=33868800, interface=interface)
        samples = chip.generate(100)
        assert samples.shape == (100, 6)


# =============================================================================
# OPLL Variant Tests
# =============================================================================


class TestYM2423:
    """Tests for YM2423 (OPLL variant) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2423, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2423(clock=3579545)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)


class TestYMF281:
    """Tests for YMF281 (OPLLP) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YMF281, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YMF281(clock=3579545)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)


class TestDS1001:
    """Tests for DS1001 (VRC7/Konami OPLL) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.DS1001, 3579545, 2)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.DS1001(clock=3579545)
        samples = chip.generate(100)
        assert samples.shape == (100, 2)


# =============================================================================
# Parametrized tests for all OPL chips
# =============================================================================


@pytest.fixture(params=list(ALL_OPL_CHIPS.keys()))
def opl_chip_name(request):
    """Fixture providing each OPL chip name."""
    return request.param


@pytest.fixture
def opl_chip(opl_chip_name):
    """Fixture providing each OPL chip instance."""
    chip_class, clock, _ = ALL_OPL_CHIPS[opl_chip_name]
    return chip_class(clock=clock)


class TestOPLCommon:
    """Common tests for all OPL family chips."""

    def test_interface_property(self, opl_chip):
        """Test interface property."""
        assert opl_chip.interface is not None
        assert isinstance(opl_chip.interface, ymfm.ChipInterface)

    def test_reset(self, opl_chip):
        """Test chip reset."""
        opl_chip.reset()
        samples = opl_chip.generate(100)
        assert samples is not None

    def test_generate_zero_samples(self, opl_chip):
        """Test generating zero samples."""
        samples = opl_chip.generate(0)
        assert samples.shape == (0, opl_chip.outputs)

    def test_state_save_restore(self, opl_chip):
        """Test state save/restore."""
        state = opl_chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0
        opl_chip.load_state(state)

    def test_long_generation(self, opl_chip):
        """Test generating many samples."""
        samples = opl_chip.generate(10000)
        assert samples.shape == (10000, opl_chip.outputs)

    def test_multiple_resets(self, opl_chip):
        """Test multiple resets."""
        for _ in range(5):
            opl_chip.reset()
            opl_chip.generate(10)
