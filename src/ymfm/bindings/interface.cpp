#include "interface.hpp"

namespace ymfm_py {

// Define the static member from SampleBuffer (common.hpp)
int32_t SampleBuffer::empty_buffer[1] = {0};

void bind_interface(py::module_& m) {
    // Bind access_class enum
    py::enum_<ymfm::access_class>(m, "AccessClass")
        .value("IO", ymfm::ACCESS_IO)
        .value("ADPCM_A", ymfm::ACCESS_ADPCM_A)
        .value("ADPCM_B", ymfm::ACCESS_ADPCM_B)
        .value("PCM", ymfm::ACCESS_PCM)
        .export_values();

    // Bind ChipInterface base class
    py::class_<ChipInterface, PyChipInterface, std::shared_ptr<ChipInterface>>(m, "ChipInterface",
        R"doc(
        Base interface class for ymfm chip callbacks.

        This class can be subclassed in Python to provide custom behavior for:
        - Timer management
        - IRQ handling
        - External memory access (for ADPCM/PCM data)

        For simple audio rendering without timing accuracy, the default
        implementation (which ignores timers and IRQs) is sufficient.
        )doc")
        .def(py::init<>())
        .def("sync_mode_write", &ChipInterface::ymfm_sync_mode_write,
            py::arg("data"),
            "Called when the mode register is written (for synchronization)")
        .def("sync_check_interrupts", &ChipInterface::ymfm_sync_check_interrupts,
            "Process pending interrupts")
        .def("set_timer", &ChipInterface::ymfm_set_timer,
            py::arg("timer_num"), py::arg("duration"),
            R"doc(
            Called when a timer should be set.

            Args:
                timer_num: Timer number (0 or 1)
                duration: Duration in chip clocks, or -1 to disable
            )doc")
        .def("set_busy_end", &ChipInterface::ymfm_set_busy_end,
            py::arg("clocks"),
            "Set the busy end time in clocks")
        .def("is_busy", &ChipInterface::ymfm_is_busy,
            "Check if the chip is currently busy")
        .def("update_irq", &ChipInterface::ymfm_update_irq,
            py::arg("asserted"),
            "Called when IRQ state changes")
        .def("external_read", &ChipInterface::ymfm_external_read,
            py::arg("access_type"), py::arg("address"),
            R"doc(
            Read from external memory (ADPCM/PCM data).

            Args:
                access_type: Type of access (AccessClass enum)
                address: Memory address to read

            Returns:
                Byte value at the address
            )doc")
        .def("external_write", &ChipInterface::ymfm_external_write,
            py::arg("access_type"), py::arg("address"), py::arg("data"),
            R"doc(
            Write to external memory.

            Args:
                access_type: Type of access (AccessClass enum)
                address: Memory address to write
                data: Byte value to write
            )doc");

    // Bind MemoryInterface with built-in memory buffer
    py::class_<MemoryInterface, ChipInterface, std::shared_ptr<MemoryInterface>>(m, "MemoryInterface",
        R"doc(
        Interface with built-in memory buffer support.

        This interface provides a simple way to supply ADPCM/PCM data
        to chips that require external memory. Set the memory buffer
        using set_memory() before generating samples.
        )doc")
        .def(py::init<>())
        .def("set_memory", &MemoryInterface::set_memory,
            py::arg("data"),
            "Set the memory buffer (bytes object)")
        .def("get_memory", &MemoryInterface::get_memory,
            "Get the current memory buffer");

    // Bind SsgOverride for custom SSG implementations
    py::class_<SsgOverride, PySsgOverride, std::shared_ptr<SsgOverride>>(m, "SsgOverride",
        R"doc(
        Interface for providing custom SSG (AY-3-8910 / YM2149) implementations.

        Some FM chips (YM2203, YM2608, YM2610, YMF288) include an SSG component.
        By default, the internal SSG emulation is used. Subclass this to provide
        a custom SSG implementation (e.g., for more accurate emulation or to
        use external hardware).

        Use chip.set_ssg_override(override) to replace the internal SSG.
        )doc")
        .def(py::init<>())
        .def("ssg_reset", &SsgOverride::ssg_reset,
            "Reset the SSG to initial state")
        .def("ssg_read", &SsgOverride::ssg_read,
            py::arg("regnum"),
            R"doc(
            Read from SSG register.

            Args:
                regnum: Register number (0-15)

            Returns:
                Byte value from the register
            )doc")
        .def("ssg_write", &SsgOverride::ssg_write,
            py::arg("regnum"), py::arg("data"),
            R"doc(
            Write to SSG register.

            Args:
                regnum: Register number (0-15)
                data: Byte value to write
            )doc")
        .def("ssg_prescale_changed", &SsgOverride::ssg_prescale_changed,
            "Called when the SSG prescaler has changed");
}

} // namespace ymfm_py
