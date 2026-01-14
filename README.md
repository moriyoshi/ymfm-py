# ymfm-py

Python bindings for the [ymfm](https://github.com/aaronsgiles/ymfm) FM synthesis library.

## Supported Chips

- **OPL Family**: YM3526 (OPL), Y8950 (MSX-Audio), YM3812 (OPL2), YMF262 (OPL3), YMF289B (OPL3L), YMF278B (OPL4)
- **OPLL Family**: YM2413 (OPLL), YM2423, YMF281 (OPLLP), DS1001 (Konami VRC7)
- **OPN Family**: YM2203 (OPN), YM2608 (OPNA), YM2610/YM2610B (OPNB), YM2612/YM3438/YMF276 (OPN2), YMF288 (OPN3L)
- **OPM Family**: YM2151 (OPM), YM2164 (OPP)
- **OPQ Family**: YM3806, YM3533
- **OPZ Family**: YM2414 (TX81Z/DX11)
- **SSG Family**: YM2149 (AY-3-8910 compatible)

## Installation

```bash
pip install ymfm-py
```

## Usage

```python
import ymfm

# Create a Sega Genesis sound chip
chip = ymfm.YM2612(clock=7670453)

# Write to registers
chip.reset()
chip.write(0, 0x22)  # Select register
chip.write(1, 0x00)  # Write value

# Generate audio samples
samples = chip.generate(1024)  # Returns numpy array
```

## Bundled Dependencies

This package includes the [ymfm](https://github.com/aaronsgiles/ymfm) FM synthesis library by Aaron Giles, which is compiled and distributed as part of ymfm-py. The ymfm source code is located in `vendor/ymfm/`.

## Examples

The `examples/` directory contains sample scripts demonstrating ymfm-py usage:

### vgmrender.py

VGM file renderer - a Python port of the vgmrender example from ymfm. Renders VGM (Video Game Music) files to WAV format with support for gzip-compressed `.vgz` files.

```bash
python examples/vgmrender.py input.vgm -o output.wav [-r 44100]
```

Supported chips: YM2149, YM2151, YM2203, YM2413, YM2608, YM2610, YM2612, YM3526, Y8950, YM3812, YMF262

### sine_waves.py

Generates pure sine waves using all supported FM chips. Demonstrates how to program each chip type's registers to generate sine waves at specific frequencies (110Hz, 220Hz, 440Hz, 880Hz). Outputs WAV files to the `output/` directory.

```bash
python examples/sine_waves.py
```

Covers: OPL, OPLL, OPN, OPM, OPQ, and OPZ families.

### csm.py

WAV to VGM converter using CSM (Composite Sinusoidal Modeling) voice synthesis. Extracts 4 formant frequencies from audio and generate a VGM file that reproduces them on FM chips using the true CSM hardware mode (with manual key-on). Based on [csm_voice_encode_synthesis_python](https://github.com/yas-sim/csm_voice_encode_synthesis_python) by Yasunori Shimura.

```bash
python examples/csm.py [-c CHIP] [-i input.wav] output.vgm
python examples/csm.py --test [-c CHIP] output.vgm  # Generate test VGM
```

Supported chips: YM2151, YM2203, YM2608, YM3526, YM3812, Y8950

## License

BSD-3-Clause

Both pymfm and the bundled ymfm library are licensed under the BSD 3-Clause License. The example code in `examples/csm.py` is derived from csm_voice_encode_synthesis_python by Yasunori Shimura under the MIT License. See the [LICENSE](LICENSE) file for details.
