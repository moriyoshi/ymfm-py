#include "interface.hpp"

namespace ymfm_py {

// Forward declarations for binding functions
void bind_interface(py::module_& m);
void bind_chips_opl(py::module_& m);
void bind_chips_opn(py::module_& m);
void bind_chips_opm(py::module_& m);
void bind_chips_opq(py::module_& m);
void bind_chips_opz(py::module_& m);
void bind_chips_ssg(py::module_& m);

} // namespace ymfm_py

PYBIND11_MODULE(_core, m) {
    m.doc() = R"doc(
ymfm - Python bindings for the ymfm FM synthesis library

This module provides Python bindings for ymfm, a collection of Yamaha FM sound
chip emulators supporting 20+ chip variants. Supported chip families include:

OPL Family (PC/DOS games):
    - YM3526 (OPL) - 9 FM channels, mono
    - Y8950 (MSX-Audio) - 9 FM channels + ADPCM
    - YM3812 (OPL2) - 9 FM channels, 4 waveforms (Sound Blaster)
    - YMF262 (OPL3) - 18 FM channels, 4-channel output
    - YMF289B (OPL3L) - OPL3 variant, stereo output
    - YMF278B (OPL4) - OPL3 + wavetable PCM, 6 outputs
    - YM2413 (OPLL) - 9 FM channels with preset instruments
    - YM2423/YMF281/DS1001 - OPLL variants

OPN Family (Console/arcade games):
    - YM2203 (OPN) - 3 FM channels + SSG
    - YM2608 (OPNA) - 6 FM channels + SSG + ADPCM (PC-98)
    - YM2610/YM2610B (OPNB) - FM + SSG + ADPCM (Neo Geo)
    - YM2612/YM3438/YMF276 (OPN2) - 6 FM channels (Sega Genesis)
    - YMF288 (OPN3L) - 6 FM + SSG + ADPCM-A

OPM Family (Arcade/synthesizers):
    - YM2151 (OPM) - 8 FM channels, stereo (Sharp X68000, arcade)
    - YM2164 (OPP) - YM2151 variant

OPQ Family (Arcade):
    - YM3806 (OPQ) - 8 FM channels, 2 operators
    - YM3533 - OPQ variant

OPZ Family (Synthesizers):
    - YM2414 (OPZ) - 8 FM channels, 4 operators (TX81Z/DX11)

SSG/PSG Family (Sound generators):
    - YM2149 (SSG/PSG) - 3 square wave channels + noise (AY-3-8910 compatible)

Basic usage:
    import ymfm

    # Create a chip instance with clock frequency
    chip = ymfm.YM2612(clock=7670453)

    # Reset and write to registers
    chip.reset()
    chip.write(0, 0x30)  # Select register
    chip.write(1, 0x7F)  # Write value

    # Generate audio samples (returns memoryview with interleaved channels)
    samples = chip.generate(1024)  # Returns memoryview of int32

For chips with ADPCM/PCM support (Y8950, YM2608, YM2610, YMF278B), use
MemoryInterface to provide sample data:

    interface = ymfm.MemoryInterface()
    interface.set_memory(adpcm_data)
    chip = ymfm.YM2608(clock=8000000, interface=interface)

For chips with SSG (YM2203, YM2608, YM2610, YMF288), you can provide a
custom SSG implementation using SsgOverride:

    class MySsg(ymfm.SsgOverride):
        def ssg_read(self, regnum): return 0
        def ssg_write(self, regnum, data): pass

    chip = ymfm.YM2203(clock=4000000)
    chip.set_ssg_override(MySsg())
)doc";

    // Bind all components
    ymfm_py::bind_interface(m);
    ymfm_py::bind_chips_opl(m);
    ymfm_py::bind_chips_opn(m);
    ymfm_py::bind_chips_opm(m);
    ymfm_py::bind_chips_opq(m);
    ymfm_py::bind_chips_opz(m);
    ymfm_py::bind_chips_ssg(m);
}
