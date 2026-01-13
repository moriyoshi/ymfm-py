#include "interface.hpp"
#include <memory>

namespace ymfm_py {

// Stub wrapper classes for OPL family chips
// These wrappers manage the interface lifetime and provide a clean Python API

// ======================> YM3526Chip (OPL)

class YM3526Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym3526::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM3526";
    static constexpr const char* CHIP_DESCRIPTION = "OPL - 9 FM channels, mono output";

    YM3526Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym3526, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym3526 m_chip;
};
constexpr const char* YM3526Chip::CHIP_DESCRIPTION;

// ======================> Y8950Chip (MSX-Audio)

class Y8950Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::y8950::OUTPUTS;
    static constexpr const char* CHIP_NAME = "Y8950";
    static constexpr const char* CHIP_DESCRIPTION = "MSX-Audio - 9 FM channels + ADPCM, mono output";

    Y8950Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<MemoryInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::y8950, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::y8950 m_chip;
};
constexpr const char* Y8950Chip::CHIP_DESCRIPTION;

// ======================> YM3812Chip (OPL2)

class YM3812Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym3812::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM3812";
    static constexpr const char* CHIP_DESCRIPTION = "OPL2 - 9 FM channels, 4 waveforms, mono output (Sound Blaster)";

    YM3812Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym3812, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym3812 m_chip;
};
constexpr const char* YM3812Chip::CHIP_DESCRIPTION;

// ======================> YMF262Chip (OPL3)

class YMF262Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ymf262::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YMF262";
    static constexpr const char* CHIP_DESCRIPTION = "OPL3 - 18 FM channels, 8 waveforms, 4-channel output (Sound Blaster Pro/16)";

    YMF262Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ymf262, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ymf262 m_chip;
};
constexpr const char* YMF262Chip::CHIP_DESCRIPTION;

// ======================> YM2413Chip (OPLL)

class YM2413Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2413::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2413";
    static constexpr const char* CHIP_DESCRIPTION = "OPLL - 9 FM channels, 15 preset instruments, mono output";

    YM2413Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2413, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym2413 m_chip;
};
constexpr const char* YM2413Chip::CHIP_DESCRIPTION;

// ======================> YMF262Chip has write_address_hi, add to other chips

// ======================> YMF289BChip (OPL3L)

class YMF289BChip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ymf289b::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YMF289B";
    static constexpr const char* CHIP_DESCRIPTION = "OPL3L - OPL3 variant with 2-channel stereo output";

    YMF289BChip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read_data() { return m_chip.read_data(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ymf289b, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ymf289b m_chip;
};
constexpr const char* YMF289BChip::CHIP_DESCRIPTION;

// ======================> YMF278BChip (OPL4)

class YMF278BChip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ymf278b::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YMF278B";
    static constexpr const char* CHIP_DESCRIPTION = "OPL4 - OPL3 + wavetable PCM, 6-channel output";

    YMF278BChip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<MemoryInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read_data_pcm() { return m_chip.read_data_pcm(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write_address_pcm(uint8_t data) { m_chip.write_address_pcm(data); }
    void write_data_pcm(uint8_t data) { m_chip.write_data_pcm(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ymf278b, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ymf278b m_chip;
};
constexpr const char* YMF278BChip::CHIP_DESCRIPTION;

// ======================> YM2423Chip (OPLL variant)

class YM2423Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2423::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2423";
    static constexpr const char* CHIP_DESCRIPTION = "OPLL variant - 9 FM channels with alternate preset instruments";

    YM2423Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2423, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym2423 m_chip;
};
constexpr const char* YM2423Chip::CHIP_DESCRIPTION;

// ======================> YMF281Chip (OPLL variant - OPLLP)

class YMF281Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ymf281::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YMF281";
    static constexpr const char* CHIP_DESCRIPTION = "OPLLP - OPLL variant with alternate preset instruments";

    YMF281Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ymf281, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ymf281 m_chip;
};
constexpr const char* YMF281Chip::CHIP_DESCRIPTION;

// ======================> DS1001Chip (OPLL variant - Konami VRC7)

class DS1001Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ds1001::OUTPUTS;
    static constexpr const char* CHIP_NAME = "DS1001";
    static constexpr const char* CHIP_DESCRIPTION = "VRC7 - Konami OPLL variant used in NES games";

    DS1001Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ds1001, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ds1001 m_chip;
};
constexpr const char* DS1001Chip::CHIP_DESCRIPTION;

// Helper macro for binding common chip methods
#define BIND_CHIP_COMMON(cls) \
    .def(py::init<uint32_t, std::shared_ptr<ChipInterface>>(), \
        py::arg("clock"), py::arg("interface") = nullptr, \
        "Create a new " #cls " chip instance.\n\n" \
        "Args:\n" \
        "    clock: Input clock frequency in Hz\n" \
        "    interface: Optional ChipInterface for callbacks") \
    .def("reset", &cls::reset, "Reset the chip to initial state") \
    .def("read_status", &cls::read_status, "Read the status register") \
    .def("read", &cls::read, py::arg("offset"), "Read from a register offset") \
    .def("write_address", &cls::write_address, py::arg("data"), "Write to the address register") \
    .def("write_data", &cls::write_data, py::arg("data"), "Write to the data register") \
    .def("write", &cls::write, py::arg("offset"), py::arg("data"), "Write to a register at offset") \
    .def("generate", &cls::generate, py::arg("num_samples") = 1, \
        "Generate audio samples.\n\n" \
        "Args:\n" \
        "    num_samples: Number of samples to generate (default: 1)\n\n" \
        "Returns:\n" \
        "    memoryview of int32 with interleaved samples (length = num_samples * outputs)") \
    .def_property_readonly("sample_rate", &cls::sample_rate, "Output sample rate in Hz") \
    .def_property_readonly("clock", &cls::clock, "Input clock frequency in Hz") \
    .def_property_readonly("outputs", [](const cls&) { return cls::OUTPUTS; }, "Number of output channels") \
    .def("save_state", &cls::save_state, "Save chip state to bytes") \
    .def("load_state", &cls::load_state, py::arg("data"), "Load chip state from bytes") \
    .def_property_readonly("interface", &cls::interface, "Get the chip interface")

// Binding function for OPL family
void bind_chips_opl(py::module_& m) {
    // YM3526 (OPL)
    py::class_<YM3526Chip>(m, "YM3526", YM3526Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM3526Chip);

    // Y8950 (MSX-Audio)
    py::class_<Y8950Chip>(m, "Y8950", Y8950Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(Y8950Chip);

    // YM3812 (OPL2)
    py::class_<YM3812Chip>(m, "YM3812", YM3812Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM3812Chip);

    // YMF262 (OPL3)
    py::class_<YMF262Chip>(m, "YMF262", YMF262Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YMF262Chip)
        .def("write_address_hi", &YMF262Chip::write_address_hi, py::arg("data"),
            "Write to the high address register (for registers 0x100+)");

    // YM2413 (OPLL)
    py::class_<YM2413Chip>(m, "YM2413", YM2413Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2413Chip);

    // YMF289B (OPL3L)
    py::class_<YMF289BChip>(m, "YMF289B", YMF289BChip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YMF289BChip)
        .def("write_address_hi", &YMF289BChip::write_address_hi, py::arg("data"),
            "Write to the high address register (for registers 0x100+)")
        .def("read_data", &YMF289BChip::read_data, "Read the data register");

    // YMF278B (OPL4)
    py::class_<YMF278BChip>(m, "YMF278B", YMF278BChip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YMF278BChip)
        .def("write_address_hi", &YMF278BChip::write_address_hi, py::arg("data"),
            "Write to the high address register (for registers 0x100+)")
        .def("write_address_pcm", &YMF278BChip::write_address_pcm, py::arg("data"),
            "Write to the PCM address register")
        .def("write_data_pcm", &YMF278BChip::write_data_pcm, py::arg("data"),
            "Write to the PCM data register")
        .def("read_data_pcm", &YMF278BChip::read_data_pcm, "Read PCM data register");

    // YM2423 (OPLL variant)
    py::class_<YM2423Chip>(m, "YM2423", YM2423Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2423Chip);

    // YMF281 (OPLLP)
    py::class_<YMF281Chip>(m, "YMF281", YMF281Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YMF281Chip);

    // DS1001 (VRC7)
    py::class_<DS1001Chip>(m, "DS1001", DS1001Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(DS1001Chip);
}

} // namespace ymfm_py
