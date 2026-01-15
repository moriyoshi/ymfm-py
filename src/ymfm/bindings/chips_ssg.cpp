#include "interface.hpp"
#include <memory>

namespace ymfm_py {

// Binding file for SSG/PSG chip family:
// - YM2149 (SSG/PSG - AY-3-8910 compatible)

// ======================> YM2149Chip (SSG/PSG)

class YM2149Chip {
public:
    static constexpr uint32_t OUTPUTS = ymfm::ym2149::OUTPUTS;
    static constexpr const char* CHIP_NAME = "YM2149";
    static constexpr const char* CHIP_DESCRIPTION = "SSG/PSG - 3 square wave channels + noise (AY-3-8910 compatible)";

    YM2149Chip(uint32_t clock, std::shared_ptr<ChipInterface> interface = nullptr)
        : m_clock(clock)
        , m_interface(interface ? interface : std::make_shared<ChipInterface>())
        , m_chip(*m_interface)
    {
        m_chip.reset();
    }

    void reset() { m_chip.reset(); }

    // Stub for API consistency - SSG chips don't have a status register
    uint8_t read_status() { return 0; }
    uint8_t read_data() { return m_chip.read_data(); }
    uint8_t read(uint32_t offset) { return m_chip.read(offset); }

    void write_address(uint8_t data) { m_chip.write_address(data); }
    void write_data(uint8_t data) { m_chip.write_data(data); }
    void write(uint32_t offset, uint8_t data) { m_chip.write(offset, data); }

    py::memoryview generate(uint32_t num_samples) {
        return generate_samples<ymfm::ym2149, OUTPUTS>(m_chip, num_samples);
    }

    uint32_t generate_into(py::buffer buffer) {
        return generate_samples_into<ymfm::ym2149, OUTPUTS>(m_chip, buffer);
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
    ymfm::ym2149 m_chip;
};
constexpr const char* YM2149Chip::CHIP_DESCRIPTION;

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

// Helper macro for chips with read_data (no status)
#define BIND_READ_DATA(cls) \
    .def("read_data", &cls::read_data, "Read the data register")

// Helper macro for SSG override support
#define BIND_SSG_OVERRIDE(cls) \
    .def("set_ssg_override", &cls::set_ssg_override, py::arg("override"), \
        "Set a custom SSG implementation to override the internal SSG.\n\n" \
        "Args:\n" \
        "    override: SsgOverride instance, or None to use internal SSG") \
    .def_property_readonly("ssg_override", &cls::ssg_override, \
        "Get the current SSG override (None if using internal SSG)")

// Binding function for SSG chip family
void bind_chips_ssg(py::module_& m) {
    // YM2149 (SSG/PSG)
    py::class_<YM2149Chip>(m, "YM2149", YM2149Chip::CHIP_DESCRIPTION)
        BIND_CHIP_COMMON(YM2149Chip)
        BIND_READ_DATA(YM2149Chip)
        BIND_SSG_OVERRIDE(YM2149Chip);
}

} // namespace ymfm_py
