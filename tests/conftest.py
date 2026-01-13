"""Shared pytest fixtures and utilities for ymfm-py tests."""

import sys
from pathlib import Path

import numpy as np

# Add examples to path for importing setup functions
sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))


def memoryview_to_numpy(samples: memoryview, num_outputs: int) -> np.ndarray:
    """Convert 2D memoryview to numpy array with shape (num_samples, num_outputs)."""
    return np.asarray(samples)


def detect_dominant_frequency(
    samples: memoryview, sample_rate: int, num_outputs: int = 2
) -> float:
    """Detect the dominant frequency in audio using FFT."""
    # Convert memoryview to numpy array
    arr = memoryview_to_numpy(samples, num_outputs)

    # Use left channel
    audio = arr[:, 0].astype(np.float64)

    # Apply window
    window = np.hanning(len(audio))
    audio = audio * window

    # FFT
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sample_rate)

    # Find peak (skip DC component)
    peak_idx = np.argmax(fft[1:]) + 1
    return freqs[peak_idx]


class ChipTestHelper:
    """Helper class for chip testing."""

    @staticmethod
    def test_basic_instantiation(chip_class, clock, expected_outputs=None):
        """Test basic chip instantiation."""
        chip = chip_class(clock=clock)
        assert chip is not None
        assert chip.clock == clock
        assert chip.sample_rate > 0
        assert chip.outputs > 0
        if expected_outputs is not None:
            assert chip.outputs == expected_outputs
        return chip

    @staticmethod
    def test_generate_samples(chip, num_samples=100):
        """Test sample generation."""
        samples = chip.generate(num_samples)
        assert isinstance(samples, memoryview)
        assert samples.shape == (num_samples, chip.outputs)
        assert samples.format == "i"  # int32
        return samples

    @staticmethod
    def test_state_roundtrip(chip):
        """Test state save/restore roundtrip."""
        # Generate some samples to advance state
        chip.generate(100)

        # Save state
        state = chip.save_state()
        assert isinstance(state, bytes)
        assert len(state) > 0

        # Generate reference samples
        samples1 = chip.generate(100)

        # Load state back
        chip.load_state(state)

        # Generate samples again
        samples2 = chip.generate(100)

        # Should be identical (compare as bytes)
        assert samples1.tobytes() == samples2.tobytes()

    @staticmethod
    def test_register_operations(chip):
        """Test register read/write operations."""
        chip.write_address(0x00)
        chip.write_data(0x00)
        chip.write(0, 0x00)
        chip.write(1, 0x00)

    @staticmethod
    def test_reset(chip):
        """Test chip reset."""
        chip.reset()
        samples = chip.generate(100)
        assert samples is not None
