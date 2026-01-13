from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ._core import ChipInterface


# Type alias for chip output (memoryview of interleaved int32 samples)
ChipOutput = memoryview


class Chip(Protocol):
    """
    Protocol defining the common interface for all ymfm chip implementations.

    All chip classes (YM2151, YM2612, YM3812, etc.) implicitly satisfy this protocol.
    """

    def reset(self) -> None:
        """Reset the chip to its initial state."""
        ...

    def read_status(self) -> int:
        """Read the chip status register."""
        ...

    def read(self, offset: int) -> int:
        """Read from a chip register."""
        ...

    def write_address(self, data: int) -> None:
        """Write to the address register."""
        ...

    def write_data(self, data: int) -> None:
        """Write to the data register."""
        ...

    def write(self, offset: int, data: int) -> None:
        """Write to a chip register (combined address/data)."""
        ...

    def generate(self, num_samples: int = 1) -> ChipOutput:
        """
        Generate audio samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            memoryview of int32 with interleaved samples (length = samples * outputs)
        """
        ...

    def save_state(self) -> bytes:
        """Save the chip state to bytes."""
        ...

    def load_state(self, data: bytes) -> None:
        """Load the chip state from bytes."""
        ...

    @property
    def sample_rate(self) -> int:
        """The chip's output sample rate in Hz."""
        ...

    @property
    def clock(self) -> int:
        """The chip's clock frequency in Hz."""
        ...

    @property
    def outputs(self) -> int:
        """Number of output channels."""
        ...

    @property
    def interface(self) -> "ChipInterface":
        """The chip interface for callbacks."""
        ...
