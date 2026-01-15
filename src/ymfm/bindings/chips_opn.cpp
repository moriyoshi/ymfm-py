#include "interface.hpp"
#include <memory>

namespace ymfm_py {

// Stub wrapper classes for OPN family chips
// These wrappers manage the interface lifetime and provide a clean Python API

// ======================> YM2203Chip (OPN)

class YM2203Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2203::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2203";
    static constexpr const char* CHIP_DESCRIPTION = "OPN - 3 FM channels + SSG, stereo output";

    YM2203Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
        return generate_samples<ymfm::ym2203, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2203, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

    // SSG override support
    void set_ssg_override(std::shared_ptr<SsgOverride> ssg) {
        m_ssg_override = ssg;
        if (ssg) {
            m_chip.ssg_override(*ssg);
        }
    }

    std::shared_ptr<SsgOverride> ssg_override() const { return m_ssg_override; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    std::shared_ptr<SsgOverride> m_ssg_override;
    ymfm::ym2203 m_chip;
};
constexpr const char* YM2203Chip::CHIP_DESCRIPTION;

// ======================> YM2608Chip (OPNA)

class YM2608Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2608::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2608";
    static constexpr const char* CHIP_DESCRIPTION = "OPNA - 6 FM channels + SSG + ADPCM, 3-channel output (PC-98)";

    YM2608Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<MemoryInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read_status_hi() { return m_chip.read_status_hi(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2608, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2608, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

    // SSG override support
    void set_ssg_override(std::shared_ptr<SsgOverride> ssg) {
        m_ssg_override = ssg;
        if (ssg) {
            m_chip.ssg_override(*ssg);
        }
    }

    std::shared_ptr<SsgOverride> ssg_override() const { return m_ssg_override; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    std::shared_ptr<SsgOverride> m_ssg_override;
    ymfm::ym2608 m_chip;
};
constexpr const char* YM2608Chip::CHIP_DESCRIPTION;

// ======================> YM2610Chip (OPNB)

class YM2610Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2610::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2610";
    static constexpr const char* CHIP_DESCRIPTION = "OPNB - 4 FM channels + SSG + ADPCM, stereo output (Neo Geo)";

    YM2610Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<MemoryInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read_status_hi() { return m_chip.read_status_hi(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2610, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2610, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

    // SSG override support
    void set_ssg_override(std::shared_ptr<SsgOverride> ssg) {
        m_ssg_override = ssg;
        if (ssg) {
            m_chip.ssg_override(*ssg);
        }
    }

    std::shared_ptr<SsgOverride> ssg_override() const { return m_ssg_override; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    std::shared_ptr<SsgOverride> m_ssg_override;
    ymfm::ym2610 m_chip;
};
constexpr const char* YM2610Chip::CHIP_DESCRIPTION;

// ======================> YM2610BChip (OPNB variant)

class YM2610BChip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2610b::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2610B";
    static constexpr const char* CHIP_DESCRIPTION = "OPNB variant - 6 FM channels + SSG + ADPCM, stereo output";

    YM2610BChip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<MemoryInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read_status_hi() { return m_chip.read_status_hi(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2610b, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2610b, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

    // SSG override support
    void set_ssg_override(std::shared_ptr<SsgOverride> ssg) {
        m_ssg_override = ssg;
        if (ssg) {
            m_chip.ssg_override(*ssg);
        }
    }

    std::shared_ptr<SsgOverride> ssg_override() const { return m_ssg_override; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    std::shared_ptr<SsgOverride> m_ssg_override;
    ymfm::ym2610b m_chip;
};
constexpr const char* YM2610BChip::CHIP_DESCRIPTION;

// ======================> YM2612Chip (OPN2 - Sega Genesis)

class YM2612Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2612::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2612";
    static constexpr const char* CHIP_DESCRIPTION = "OPN2 - 6 FM channels, stereo output (Sega Genesis/Mega Drive)";

    YM2612Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2612, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2612, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym2612 m_chip;
};
constexpr const char* YM2612Chip::CHIP_DESCRIPTION;

// ======================> YM3438Chip (OPN2C - Genesis variant)

class YM3438Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym3438::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM3438";
    static constexpr const char* CHIP_DESCRIPTION = "OPN2C - YM2612 variant with improved DAC, stereo output";

    YM3438Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym3438, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym3438, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym3438 m_chip;
};
constexpr const char* YM3438Chip::CHIP_DESCRIPTION;

// ======================> YMF276Chip (OPN2 variant)

class YMF276Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ymf276::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YMF276";
    static constexpr const char* CHIP_DESCRIPTION = "OPN2 variant - YM2612 with improved DAC accuracy";

    YMF276Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ymf276, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ymf276, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ymf276 m_chip;
};
constexpr const char* YMF276Chip::CHIP_DESCRIPTION;

// ======================> YMF288Chip (OPN3L)

class YMF288Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ymf288::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YMF288";
    static constexpr const char* CHIP_DESCRIPTION = "OPN3L - 6 FM + SSG + ADPCM-A, 3-channel output";

    YMF288Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<MemoryInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read_status_hi() { return m_chip.read_status_hi(); }
    uint8_t read_data() { return m_chip.read_data(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write_address_hi(uint8_t data) { m_chip.write_address_hi(data); }
    void write_data_hi(uint8_t data) { m_chip.write_data_hi(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ymf288, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ymf288, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

    // SSG override support
    void set_ssg_override(std::shared_ptr<SsgOverride> ssg) {
        m_ssg_override = ssg;
        if (ssg) {
            m_chip.ssg_override(*ssg);
        }
    }

    std::shared_ptr<SsgOverride> ssg_override() const { return m_ssg_override; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    std::shared_ptr<SsgOverride> m_ssg_override;
    ymfm::ymf288 m_chip;
};
constexpr const char* YMF288Chip::CHIP_DESCRIPTION;

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
    .def("generate_into", &cls::generate_into, py::arg("buffer"), \
        "Generate audio samples into a provided buffer.\n\n" \
        "Args:\n" \
        "    buffer: Writable buffer with int32 dtype. Can be 1D (length must be\n" \
        "            divisible by outputs) or 2D with shape (N, outputs).\n\n" \
        "Returns:\n" \
        "    Number of samples generated") \
    .def_property_readonly("sample_rate", &cls::sample_rate, "Output sample rate in Hz") \
    .def_property_readonly("clock", &cls::clock, "Input clock frequency in Hz") \
    .def_property_readonly("outputs", [](const cls&) { return cls::OUTPUTS; }, "Number of output channels") \
    .def("save_state", &cls::save_state, "Save chip state to bytes") \
    .def("load_state", &cls::load_state, py::arg("data"), "Load chip state from bytes") \
    .def_property_readonly("interface", &cls::interface, "Get the chip interface")

// Helper macro for OPN chips with high address register
#define BIND_OPN_HI_ADDR(cls) \
    .def("write_address_hi", &cls::write_address_hi, py::arg("data"), \
        "Write to the high address register (for registers 0x100+)") \
    .def("write_data_hi", &cls::write_data_hi, py::arg("data"), \
        "Write to the high data register")

// Helper macro for OPN chips with high status register
#define BIND_OPN_HI_STATUS(cls) \
    .def("read_status_hi", &cls::read_status_hi, "Read the high status register")

// Helper macro for OPN chips with SSG override support
#define BIND_OPN_SSG_OVERRIDE(cls) \
    .def("set_ssg_override", &cls::set_ssg_override, py::arg("override"), \
        "Set a custom SSG implementation to override the internal SSG.\n\n" \
        "Args:\n" \
        "    override: SsgOverride instance, or None to use internal SSG") \
    .def_property_readonly("ssg_override", &cls::ssg_override, \
        "Get the current SSG override (None if using internal SSG)")

// Binding function for OPN family
void bind_chips_opn(py::module_& m) {
    // YM2203 (OPN)
    py::class_<YM2203Chip>(m, "YM2203", YM2203Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2203Chip)
        BIND_OPN_SSG_OVERRIDE(YM2203Chip);

    // YM2608 (OPNA)
    py::class_<YM2608Chip>(m, "YM2608", YM2608Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2608Chip)
        BIND_OPN_HI_ADDR(YM2608Chip)
        BIND_OPN_HI_STATUS(YM2608Chip)
        BIND_OPN_SSG_OVERRIDE(YM2608Chip);

    // YM2610 (OPNB)
    py::class_<YM2610Chip>(m, "YM2610", YM2610Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2610Chip)
        BIND_OPN_HI_ADDR(YM2610Chip)
        BIND_OPN_HI_STATUS(YM2610Chip)
        BIND_OPN_SSG_OVERRIDE(YM2610Chip);

    // YM2610B (OPNB variant)
    py::class_<YM2610BChip>(m, "YM2610B", YM2610BChip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2610BChip)
        BIND_OPN_HI_ADDR(YM2610BChip)
        BIND_OPN_HI_STATUS(YM2610BChip)
        BIND_OPN_SSG_OVERRIDE(YM2610BChip);

    // YM2612 (OPN2 - Sega Genesis)
    py::class_<YM2612Chip>(m, "YM2612", YM2612Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2612Chip)
        BIND_OPN_HI_ADDR(YM2612Chip);

    // YM3438 (OPN2C)
    py::class_<YM3438Chip>(m, "YM3438", YM3438Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM3438Chip)
        BIND_OPN_HI_ADDR(YM3438Chip);

    // YMF276 (OPN2 variant)
    py::class_<YMF276Chip>(m, "YMF276", YMF276Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YMF276Chip)
        BIND_OPN_HI_ADDR(YMF276Chip);

    // YMF288 (OPN3L)
    py::class_<YMF288Chip>(m, "YMF288", YMF288Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YMF288Chip)
        BIND_OPN_HI_ADDR(YMF288Chip)
        BIND_OPN_HI_STATUS(YMF288Chip)
        BIND_OPN_SSG_OVERRIDE(YMF288Chip)
        .def("read_data", &YMF288Chip::read_data, "Read the data register");
}

} // namespace ymfm_py
