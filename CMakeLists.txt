cmake_minimum_required(VERSION 3.15...3.27)
project(ymfm-py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find pybind11
find_package(pybind11 CONFIG REQUIRED)

# ymfm source files
set(YMFM_DIR ${CMAKE_SOURCE_DIR}/vendor/ymfm/src)
set(YMFM_SOURCES
    ${YMFM_DIR}/ymfm_adpcm.cpp
    ${YMFM_DIR}/ymfm_misc.cpp
    ${YMFM_DIR}/ymfm_opl.cpp
    ${YMFM_DIR}/ymfm_opm.cpp
    ${YMFM_DIR}/ymfm_opn.cpp
    ${YMFM_DIR}/ymfm_opq.cpp
    ${YMFM_DIR}/ymfm_opz.cpp
    ${YMFM_DIR}/ymfm_pcm.cpp
    ${YMFM_DIR}/ymfm_ssg.cpp
)

# Binding source files
set(BINDING_SOURCES
    src/ymfm/bindings/module.cpp
    src/ymfm/bindings/interface.cpp
    src/ymfm/bindings/chips_opl.cpp
    src/ymfm/bindings/chips_opn.cpp
    src/ymfm/bindings/chips_opm.cpp
    src/ymfm/bindings/chips_opq.cpp
    src/ymfm/bindings/chips_opz.cpp
    src/ymfm/bindings/chips_ssg.cpp
)

# Create the Python module
pybind11_add_module(_core MODULE
    ${YMFM_SOURCES}
    ${BINDING_SOURCES}
)

target_include_directories(_core PRIVATE
    ${YMFM_DIR}
    ${CMAKE_SOURCE_DIR}/src/ymfm/bindings
)

# Install the module to the ymfm package
install(TARGETS _core DESTINATION ymfm)
