"""Tests for SSG family chips.

SSG (Software-controlled Sound Generator) chips include:
- YM2149 (SSG/PSG) - 3 square wave channels + noise (AY-3-8910 compatible)
"""

from conftest import ChipTestHelper

import ymfm

# =============================================================================
# YM2149 (SSG/PSG) Tests
# =============================================================================


class TestYM2149:
    """Tests for YM2149 (SSG/PSG) chip."""

    def test_create(self):
        """Test chip creation."""
        ChipTestHelper.test_basic_instantiation(ymfm.YM2149, 2000000, 3)

    def test_generate(self):
        """Test sample generation."""
        chip = ymfm.YM2149(clock=2000000)
        samples = chip.generate(100)
        assert isinstance(samples, memoryview)
        assert samples.shape == (100, 3)
        assert samples.format == "i"

    def test_read_data(self):
        """Test data register read."""
        chip = ymfm.YM2149(clock=2000000)
        chip.write_address(0x07)  # Mixer control register
        data = chip.read_data()
        assert isinstance(data, int)

    def test_write_operations(self):
        """Test write operations."""
        chip = ymfm.YM2149(clock=2000000)
        chip.write_address(0x00)  # Channel A fine tune
        chip.write_data(0x55)
        chip.write(0, 0x01)  # Offset 0 sets address
        chip.write(1, 0xAA)  # Offset 1 sets data

    def test_channel_a_tone(self):
        """Test Channel A tone generation."""
        chip = ymfm.YM2149(clock=2000000)
        chip.reset()
        # Set Channel A frequency (fine + coarse tune)
        chip.write(0, 0x00)
        chip.write(1, 0x80)  # Fine tune
        chip.write(0, 0x01)
        chip.write(1, 0x00)  # Coarse tune
        # Set Channel A volume
        chip.write(0, 0x08)
        chip.write(1, 0x0F)  # Max volume
        # Enable tone on Channel A, disable noise
        chip.write(0, 0x07)
        chip.write(1, 0x3E)  # Tone A on, all else off
        samples = chip.generate(100)
        assert samples is not None

    def test_channel_b_tone(self):
        """Test Channel B tone generation."""
        chip = ymfm.YM2149(clock=2000000)
        chip.reset()
        # Set Channel B frequency
        chip.write(0, 0x02)
        chip.write(1, 0x80)  # Fine tune
        chip.write(0, 0x03)
        chip.write(1, 0x00)  # Coarse tune
        # Set Channel B volume
        chip.write(0, 0x09)
        chip.write(1, 0x0F)
        # Enable tone on Channel B
        chip.write(0, 0x07)
        chip.write(1, 0x3D)
        samples = chip.generate(100)
        assert samples is not None

    def test_channel_c_tone(self):
        """Test Channel C tone generation."""
        chip = ymfm.YM2149(clock=2000000)
        chip.reset()
        # Set Channel C frequency
        chip.write(0, 0x04)
        chip.write(1, 0x80)  # Fine tune
        chip.write(0, 0x05)
        chip.write(1, 0x00)  # Coarse tune
        # Set Channel C volume
        chip.write(0, 0x0A)
        chip.write(1, 0x0F)
        # Enable tone on Channel C
        chip.write(0, 0x07)
        chip.write(1, 0x3B)
        samples = chip.generate(100)
        assert samples is not None

    def test_noise_generator(self):
        """Test noise generator."""
        chip = ymfm.YM2149(clock=2000000)
        chip.reset()
        # Set noise period
        chip.write(0, 0x06)
        chip.write(1, 0x10)
        # Set Channel A volume
        chip.write(0, 0x08)
        chip.write(1, 0x0F)
        # Enable noise on Channel A
        chip.write(0, 0x07)
        chip.write(1, 0x37)  # Noise A on
        samples = chip.generate(100)
        assert samples is not None

    def test_envelope_generator(self):
        """Test envelope generator."""
        chip = ymfm.YM2149(clock=2000000)
        chip.reset()
        # Set envelope period
        chip.write(0, 0x0B)
        chip.write(1, 0x00)  # Fine
        chip.write(0, 0x0C)
        chip.write(1, 0x10)  # Coarse
        # Set envelope shape
        chip.write(0, 0x0D)
        chip.write(1, 0x0E)  # Continue, attack, alternate
        # Use envelope for Channel A volume
        chip.write(0, 0x08)
        chip.write(1, 0x10)  # Envelope mode
        # Enable tone on Channel A
        chip.write(0, 0x07)
        chip.write(1, 0x3E)
        samples = chip.generate(100)
        assert samples is not None

    def test_ssg_override_support(self):
        """Test SSG override support."""
        chip = ymfm.YM2149(clock=2000000)
        assert chip.ssg_override is None

        override = ymfm.SsgOverride()
        chip.set_ssg_override(override)
        assert chip.ssg_override is override

    def test_custom_ssg_override(self):
        """Test custom SSG override implementation."""

        class CustomSsg(ymfm.SsgOverride):
            def __init__(self):
                super().__init__()
                self.registers = [0] * 16
                self.reset_count = 0
                self.write_count = 0
                self.read_count = 0

            def ssg_reset(self):
                self.reset_count += 1
                self.registers = [0] * 16

            def ssg_read(self, regnum):
                self.read_count += 1
                return self.registers[regnum] if regnum < 16 else 0

            def ssg_write(self, regnum, data):
                self.write_count += 1
                if regnum < 16:
                    self.registers[regnum] = data

            def ssg_prescale_changed(self):
                pass

        custom = CustomSsg()
        chip = ymfm.YM2149(clock=2000000)
        chip.set_ssg_override(custom)

        chip.reset()
        samples = chip.generate(100)
        assert samples.shape == (100, 3)

    def test_state_save_restore(self):
        """Test state save/restore."""
        chip = ymfm.YM2149(clock=2000000)
        chip.write(0, 0x00)
        chip.write(1, 0x55)
        chip.generate(10)

        state = chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0

        chip2 = ymfm.YM2149(clock=2000000)
        chip2.load_state(state)

    def test_interface_property(self):
        """Test interface property."""
        chip = ymfm.YM2149(clock=2000000)
        assert chip.interface is not None
        assert isinstance(chip.interface, ymfm.ChipInterface)

    def test_generate_zero_samples(self):
        """Test generating zero samples."""
        chip = ymfm.YM2149(clock=2000000)
        samples = chip.generate(0)
        assert samples.shape == (0, 3)

    def test_long_generation(self):
        """Test generating many samples."""
        chip = ymfm.YM2149(clock=2000000)
        samples = chip.generate(10000)
        assert samples.shape == (10000, 3)

    def test_multiple_resets(self):
        """Test multiple resets."""
        chip = ymfm.YM2149(clock=2000000)
        for _ in range(5):
            chip.reset()
            chip.generate(10)
