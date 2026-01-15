#include "interface.hpp"
#include <memory>

namespace ymfm_py {

// Binding file for OPQ chip family:
// - YM3806 (OPQ - arcade)
// - YM3533 (OPQ variant - arcade)

// ======================> YM3806Chip (OPQ)

class YM3806Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym3806::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM3806";
    static constexpr const char* CHIP_DESCRIPTION = "OPQ - 8 FM channels, 2 operators (arcade)";

    YM3806Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    uint8_t read_status() { return m_chip.read_status(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    // Note: YM3806 only supports direct register writes, not address/data separation
    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym3806, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym3806, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym3806 m_chip;
};
constexpr const char* YM3806Chip::CHIP_DESCRIPTION;

// ======================> YM3533Chip (OPQ variant)

class YM3533Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym3533::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM3533";
    static constexpr const char* CHIP_DESCRIPTION = "OPQ variant - 8 FM channels, 2 operators (arcade)";

    YM3533Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
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
        return generate_samples<ymfm::ym3533, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym3533, OUTPUTS>(m_chip, buffer);
    }

    uint32_t sample_rate() const { return m_chip.sample_rate(m_clock); }
    uint32_t clock() const { return m_clock; }

    py::bytes save_state() { return save_chip_state(m_chip); }
    void load_state(py::bytes data) { load_chip_state(m_chip, data); }

    std::shared_ptr<ChipInterface> interface() const { return m_interface; }

private:
    uint32_t m_clock;
    std::shared_ptr<ChipInterface> m_interface;
    ymfm::ym3533 m_chip;
};
constexpr const char* YM3533Chip::CHIP_DESCRIPTION;

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

// Binding function for OPQ chip family
void bind_chips_opq(py::module_& m) {
    // YM3806 (OPQ)
    py::class_<YM3806Chip>(m, "YM3806", YM3806Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM3806Chip);

    // YM3533 (OPQ variant)
    py::class_<YM3533Chip>(m, "YM3533", YM3533Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM3533Chip);
}

} // namespace ymfm_py
