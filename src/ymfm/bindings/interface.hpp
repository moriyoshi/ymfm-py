#ifndef PYMFM_INTERFACE_HPP
#define PYMFM_INTERFACE_HPP

#include "common.hpp"

namespace ymfm_py {

// ChipInterface: A concrete implementation of ymfm_interface
// This class can be subclassed in Python to provide custom callbacks
class ChipInterface : public ymfm::ymfm_interface {
public:
    ChipInterface() = default;
    virtual ~ChipInterface() = default;

    // Timer callback - called when a timer should be set
    // tnum: timer number (0 or 1)
    // duration: timer duration in clocks, or -1 to disable
    virtual void ymfm_sync_mode_write(uint8_t data) override {
        // Call base class to actually write the mode register
        ymfm::ymfm_interface::ymfm_sync_mode_write(data);
    }

    virtual void ymfm_sync_check_interrupts() override {
        // Call base class to process interrupts
        ymfm::ymfm_interface::ymfm_sync_check_interrupts();
    }

    virtual void ymfm_set_timer(uint32_t tnum, int32_t duration) override {
        // Default: ignore timers (suitable for simple rendering)
    }

    virtual void ymfm_set_busy_end(uint32_t clocks) override {
        // Default: ignore busy state
    }

    virtual bool ymfm_is_busy() override {
        // Default: never busy
        return false;
    }

    virtual void ymfm_update_irq(bool asserted) override {
        // Default: ignore IRQ
    }

    virtual uint8_t ymfm_external_read(ymfm::access_class type, uint32_t address) override {
        // Default: return 0 for external reads
        return 0;
    }

    virtual void ymfm_external_write(ymfm::access_class type, uint32_t address, uint8_t data) override {
        // Default: ignore external writes
    }
};

// PyChipInterface: Trampoline class for Python subclassing
class PyChipInterface : public ChipInterface {
public:
    using ChipInterface::ChipInterface;

    void ymfm_sync_mode_write(uint8_t data) override {
        PYBIND11_OVERRIDE(void, ChipInterface, ymfm_sync_mode_write, data);
    }

    void ymfm_sync_check_interrupts() override {
        PYBIND11_OVERRIDE(void, ChipInterface, ymfm_sync_check_interrupts);
    }

    void ymfm_set_timer(uint32_t tnum, int32_t duration) override {
        PYBIND11_OVERRIDE(void, ChipInterface, ymfm_set_timer, tnum, duration);
    }

    void ymfm_set_busy_end(uint32_t clocks) override {
        PYBIND11_OVERRIDE(void, ChipInterface, ymfm_set_busy_end, clocks);
    }

    bool ymfm_is_busy() override {
        PYBIND11_OVERRIDE(bool, ChipInterface, ymfm_is_busy);
    }

    void ymfm_update_irq(bool asserted) override {
        PYBIND11_OVERRIDE(void, ChipInterface, ymfm_update_irq, asserted);
    }

    uint8_t ymfm_external_read(ymfm::access_class type, uint32_t address) override {
        PYBIND11_OVERRIDE(uint8_t, ChipInterface, ymfm_external_read, type, address);
    }

    void ymfm_external_write(ymfm::access_class type, uint32_t address, uint8_t data) override {
        PYBIND11_OVERRIDE(void, ChipInterface, ymfm_external_write, type, address, data);
    }
};

// MemoryInterface: Interface with memory buffer support for ADPCM/PCM
class MemoryInterface : public ChipInterface {
public:
    MemoryInterface() = default;
    virtual ~MemoryInterface() = default;

    // Set memory buffer for external reads
    void set_memory(py::bytes data) {
        std::string str = data;
        m_memory.assign(str.begin(), str.end());
    }

    // Get current memory buffer
    py::bytes get_memory() const {
        return py::bytes(reinterpret_cast<const char*>(m_memory.data()), m_memory.size());
    }

    uint8_t ymfm_external_read(ymfm::access_class type, uint32_t address) override {
        if (address < m_memory.size()) {
            return m_memory[address];
        }
        return 0;
    }

    void ymfm_external_write(ymfm::access_class type, uint32_t address, uint8_t data) override {
        if (address < m_memory.size()) {
            m_memory[address] = data;
        }
    }

private:
    std::vector<uint8_t> m_memory;
};

// SsgOverride: A concrete implementation of ssg_override for Python subclassing
// This allows users to provide custom SSG (AY-3-8910 / YM2149) implementations
class SsgOverride : public ymfm::ssg_override {
public:
    SsgOverride() = default;
    virtual ~SsgOverride() = default;

    // Reset the SSG
    virtual void ssg_reset() override {
        // Default: do nothing
    }

    // Read from SSG register
    virtual uint8_t ssg_read(uint32_t regnum) override {
        // Default: return 0
        return 0;
    }

    // Write to SSG register
    virtual void ssg_write(uint32_t regnum, uint8_t data) override {
        // Default: do nothing
    }

    // Notification when prescale has changed
    virtual void ssg_prescale_changed() override {
        // Default: do nothing
    }
};

// PySsgOverride: Trampoline class for Python subclassing
class PySsgOverride : public SsgOverride {
public:
    using SsgOverride::SsgOverride;

    void ssg_reset() override {
        PYBIND11_OVERRIDE(void, SsgOverride, ssg_reset);
    }

    uint8_t ssg_read(uint32_t regnum) override {
        PYBIND11_OVERRIDE(uint8_t, SsgOverride, ssg_read, regnum);
    }

    void ssg_write(uint32_t regnum, uint8_t data) override {
        PYBIND11_OVERRIDE(void, SsgOverride, ssg_write, regnum, data);
    }

    void ssg_prescale_changed() override {
        PYBIND11_OVERRIDE(void, SsgOverride, ssg_prescale_changed);
    }
};

// Declare the binding function
void bind_interface(py::module_& m);

} // namespace ymfm_py

#endif // PYMFM_INTERFACE_HPP
