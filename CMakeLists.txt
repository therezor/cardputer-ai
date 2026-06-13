cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(cardputer_ai)

# m5unified 0.2.17 trips a -Werror=maybe-uninitialized false positive in
# I2C_Class.hpp under some IDF 5.5 toolchains; keep it a warning there.
idf_component_get_property(m5unified_lib m5stack__m5unified COMPONENT_LIB)
target_compile_options(${m5unified_lib} PRIVATE -Wno-error=maybe-uninitialized)
