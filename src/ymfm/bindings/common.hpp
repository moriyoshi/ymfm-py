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

template<int NumOutputs, typename T, typename To> struct Drainer;

// Structure to hold buffer data and metadata together
// This ensures shape/strides arrays live as long as the buffer
template<int NumOutputs> struct SampleBuffer {
    std::vector<int32_t> data;
    Py_ssize_t shape[2];
    Py_ssize_t strides[2];

    // Create a 2D buffer with shape (num_samples, num_outputs)
    SampleBuffer(size_t num_samples, size_t num_outputs)
        : data(num_samples * num_outputs)
        , shape{static_cast<Py_ssize_t>(num_samples), static_cast<Py_ssize_t>(NumOutputs)}
        , strides{static_cast<Py_ssize_t>(NumOutputs * sizeof(int32_t)), sizeof(int32_t)}
    {}

    // Get buffer pointer, returns valid pointer even for empty buffer
    int32_t* buf() {
        // Static dummy buffer for empty case - ensures we always have a valid pointer
        static int32_t empty_buffer[1];
        return data.empty() ? empty_buffer : data.data();
    }

    template<typename T> void drain(T& chip) {
        Drainer<NumOutputs, T, typename T::output_data>(*this)(chip);
    }
};

// Generic version of Drainer class
template<int NumOutputs, typename T, typename To>
struct Drainer {
    typedef SampleBuffer<NumOutputs> inner_type;
    inner_type& inner;

    void operator()(T& chip);

    Drainer(inner_type& inner): inner(inner) {}
};

template<int NumOutputs, typename T, typename To>
inline void Drainer<NumOutputs, T, To>::operator()(T& chip) {
    To output;
    int32_t* data_ptr = inner.data.data();
    const size_t num_samples = inner.shape[0];
    const size_t num_outputs = NumOutputs;
    for (size_t i = 0; i < num_samples; ++i) {
        To output;
        chip.generate(&output);
        for (size_t ch = 0; ch < num_outputs; ++ch) {
            data_ptr[i * NumOutputs + ch] = output.data[ch];
        }
    }
}

// Specialization: direct copy to the buffer
template<int NumOutputs, typename T>
struct Drainer<NumOutputs, T, ymfm::ymfm_output<NumOutputs>> {
    typedef SampleBuffer<NumOutputs> inner_type;
    inner_type& inner;

    void operator()(T& chip);

    Drainer(inner_type& inner): inner(inner) {}
};

template<int NumOutputs, typename T>
inline void Drainer<NumOutputs, T, ymfm::ymfm_output<NumOutputs>>::operator()(T& chip) {
    chip.generate(reinterpret_cast<ymfm::ymfm_output<NumOutputs>*>(inner.data.data()), inner.shape[0]);
}

// Template function to generate samples and return as memoryview
// Returns a 2D memoryview with shape (num_samples, num_outputs)
// Data is stored in row-major (C) order: [ch0_s0, ch1_s0, ch0_s1, ch1_s1, ...]
// Releases the GIL during sample generation for better multi-threading performance
// NumOutputs is the number of output channels to expose (may differ from chip's internal output)
template<typename ChipType, size_t NumOutputs>
py::memoryview generate_samples(ChipType& chip, uint32_t num_samples) {
    // Allocate buffer structure (holds data + shape/strides)
    // SampleBuffer::buf() handles the empty case by returning a valid pointer
    auto* buffer = new SampleBuffer<NumOutputs>(num_samples, NumOutputs);

    // Release GIL during sample generation for better multi-threading
    if (num_samples > 0) {
        py::gil_scoped_release release;
        buffer->drain(chip);
    }

    // Create capsule that owns the buffer structure
    py::capsule owner(buffer, [](void* p) {
        delete reinterpret_cast<SampleBuffer<NumOutputs>*>(p);
    });

    // Use Python C-API directly to create memoryview with proper ownership
    // pybind11's from_buffer doesn't support setting the owning object
    Py_buffer view;
    view.buf = buffer->buf();
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

    PyObject* mv = PyMemoryView_FromBuffer(&view);
    if (!mv) {
        throw py::error_already_set();
    }

    // PyMemoryView_FromBuffer will incref view.obj
    Py_INCREF(view.obj);

    // Take ownership of the memoryview (steal reference)
    return py::reinterpret_steal<py::memoryview>(py::handle(mv));
}

// Template function to generate samples into a user-provided buffer
// Accepts any object implementing the buffer protocol (numpy arrays, bytearrays, etc.)
// Buffer must be writable and contain int32 data with shape (num_samples, num_outputs)
// or be a 1D buffer with length num_samples * num_outputs
// Returns the number of samples actually generated
template<typename ChipType, size_t NumOutputs>
uint32_t generate_samples_into(ChipType& chip, py::buffer buffer) {
    // Request buffer info with write access
    py::buffer_info info = buffer.request(true);  // true = writable

    // Validate format - must be int32
    if (info.format != py::format_descriptor<int32_t>::format() && info.format != "i") {
        throw py::value_error("Buffer must have int32 format (format code 'i')");
    }

    // Validate item size
    if (info.itemsize != sizeof(int32_t)) {
        throw py::value_error("Buffer item size must be 4 bytes (int32)");
    }

    // Calculate number of samples based on buffer shape
    size_t num_samples = 0;

    if (info.ndim == 1) {
        // 1D buffer: length must be divisible by num_outputs
        if (info.shape[0] % NumOutputs != 0) {
            throw py::value_error("1D buffer length must be divisible by outputs (" +
                                  std::to_string(NumOutputs) + ")");
        }
        // Check for contiguous buffer
        if (info.strides[0] != sizeof(int32_t)) {
            throw py::value_error("Buffer must be contiguous");
        }
        num_samples = info.shape[0] / NumOutputs;
    } else if (info.ndim == 2) {
        // 2D buffer: shape must be (N, num_outputs)
        if (static_cast<size_t>(info.shape[1]) != NumOutputs) {
            throw py::value_error("Buffer second dimension must equal outputs (" +
                                  std::to_string(NumOutputs) + "), got " +
                                  std::to_string(info.shape[1]));
        }
        // Check for C-contiguous (row-major) layout
        if (info.strides[1] != sizeof(int32_t) ||
            info.strides[0] != static_cast<Py_ssize_t>(NumOutputs * sizeof(int32_t))) {
            throw py::value_error("Buffer must be C-contiguous (row-major)");
        }
        num_samples = info.shape[0];
    } else {
        throw py::value_error("Buffer must be 1D or 2D");
    }

    if (num_samples == 0) {
        return 0;
    }

    // Release GIL during sample generation for better multi-threading
    // Since the buffer is validated to have the correct shape (N, NumOutputs) and is
    // C-contiguous, we can directly cast to the chip's output_data type and call generate
    {
        py::gil_scoped_release release;
        chip.generate(reinterpret_cast<typename ChipType::output_data*>(info.ptr), num_samples);
    }

    return static_cast<uint32_t>(num_samples);
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
