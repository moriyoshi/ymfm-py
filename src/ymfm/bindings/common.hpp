#ifndef PYMFM_COMMON_HPP
#define PYMFM_COMMON_HPP

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ymfm.h"
#include "ymfm_opl.h"
#include "ymfm_opm.h"
#include "ymfm_opn.h"
#include "ymfm_opq.h"
#include "ymfm_opz.h"
#include "ymfm_misc.h"
#include "ymfm_ssg.h"

#include <vector>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace ymfm_py {

// Forward declarations
class ChipInterface;

// Structure to hold buffer data and metadata together
// This ensures shape/strides arrays live as long as the buffer
struct SampleBuffer {
    std::vector<int32_t> data;
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];

    // Static dummy buffer for empty case - ensures we always have a valid pointer
    static int32_t empty_buffer[1];

    // Create a 2D buffer with shape (num_samples, num_outputs)
    SampleBuffer(size_t num_samples, size_t num_outputs)
        : data(num_samples * num_outputs)
        , shape{static_cast<Py_ssize_t>(num_samples), static_cast<Py_ssize_t>(num_outputs)}
        , strides{static_cast<Py_ssize_t>(num_outputs * sizeof(int32_t)), sizeof(int32_t)}
    {}

    // Get buffer pointer, returns valid pointer even for empty buffer
    int32_t* buf() {
        return data.empty() ? empty_buffer : data.data();
    }
};

// Static member declared here, defined in interface.cpp

// Template function to generate samples and return as memoryview
// Returns a 2D memoryview with shape (num_samples, num_outputs)
// Data is stored in row-major (C) order: [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
// Releases the GIL during sample generation for better multi-threading performance
// NumOutputs is the number of output channels to expose (may differ from chip's internal output)
template<typename ChipType, int NumOutputs>
py::memoryview generate_samples(ChipType& chip, uint32_t num_samples) {
    // Allocate buffer structure (holds data + shape/strides)
    // SampleBuffer::buf() handles the empty case by returning a valid pointer
    auto* buffer = new SampleBuffer(num_samples, NumOutputs);
    int32_t* data_ptr = buffer->buf();

    // Release GIL during sample generation for better multi-threading
    if (num_samples > 0) {
        py::gil_scoped_release release;

        // Use the chip's native output type
        typename ChipType::output_data output;
        for (uint32_t i = 0; i < num_samples; ++i) {
            chip.generate(&output);
            for (int ch = 0; ch < NumOutputs; ++ch) {
                data_ptr[i * NumOutputs + ch] = output.data[ch];
            }
        }
    }

    // Create capsule that owns the buffer structure
    py::capsule owner(buffer, [](void* p) {
        delete reinterpret_cast<SampleBuffer*>(p);
    });

    // Use Python C-API directly to create memoryview with proper ownership
    // pybind11's from_buffer doesn't support setting the owning object
    Py_buffer view;
    view.buf = data_ptr;
    view.obj = owner.ptr();  // Set the owning object
    view.len = static_cast<Py_ssize_t>(num_samples * NumOutputs * sizeof(int32_t));
    view.itemsize = sizeof(int32_t);
    view.readonly = 0;
    view.ndim = 2;
    view.format = const_cast<char*>("i");  // int32
    view.shape = buffer->shape;      // Points to buffer's shape array
    view.strides = buffer->strides;  // Points to buffer's strides array
    view.suboffsets = nullptr;
    view.internal = buffer;  // Store pointer for reference (not strictly needed but documents ownership)

    // PyMemoryView_FromBuffer will incref view.obj
    Py_INCREF(view.obj);

    PyObject* mv = PyMemoryView_FromBuffer(&view);
    if (!mv) {
        throw py::error_already_set();
    }

    // Take ownership of the memoryview (steal reference)
    return py::reinterpret_steal<py::memoryview>(py::handle(mv));
}

// Template function to save chip state to bytes
template<typename ChipType>
py::bytes save_chip_state(ChipType& chip) {
    std::vector<uint8_t> buffer;
    ymfm::ymfm_saved_state state(buffer, true);  // true = saving
    chip.save_restore(state);
    return py::bytes(reinterpret_cast<const char*>(buffer.data()), buffer.size());
}

// Template function to load chip state from bytes
template<typename ChipType>
void load_chip_state(ChipType& chip, py::bytes data) {
    std::string str = data;
    std::vector<uint8_t> buffer(str.begin(), str.end());
    ymfm::ymfm_saved_state state(buffer, false);  // false = loading
    chip.save_restore(state);
}

} // namespace ymfm_py

#endif // PYMFM_COMMON_HPP
