#include "interface.hpp"
#include <memory>

namespace ymfm_py {

// Stub wrapper classes for OPM family chips
// These wrappers manage the interface lifetime and provide a clean Python API

// ======================> YM2151Chip (OPM)

class YM2151Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2151::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2151";
    static constexpr const char* CHIP_DESCRIPTION = "OPM - 8 FM channels, 4 operators each, stereo output (Sharp X68000, arcade)";

    YM2151Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
        return generate_samples<ymfm::ym2151, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2151, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym2151 m_chip;
};
constexpr const char* YM2151Chip::CHIP_DESCRIPTION;

// ======================> YM2164Chip (OPP)

class YM2164Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2164::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2164";
    static constexpr const char* CHIP_DESCRIPTION = "OPP - YM2151 variant with half-speed Timer B, stereo output";

    YM2164Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
        return generate_samples<ymfm::ym2164, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2164, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym2164 m_chip;
};
constexpr const char* YM2164Chip::CHIP_DESCRIPTION;

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

// Binding function for OPM family
void bind_chips_opm(py::module_& m) {
    // YM2151 (OPM)
    py::class_<YM2151Chip>(m, "YM2151", YM2151Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2151Chip);

    // YM2164 (OPP)
    py::class_<YM2164Chip>(m, "YM2164", YM2164Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2164Chip);
}

} // namespace ymfm_py
