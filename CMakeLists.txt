cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(blockthetweet)

# Torch
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libs/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# SQLite3
find_library(SQLITE3_LIB sqlite3 HINTS "${CMAKE_SOURCE_DIR}/libs/sqlite3" REQUIRED)
include_directories("${CMAKE_SOURCE_DIR}/libs/sqlite3")

# xxhash
add_executable(blockthetweet src/main.cpp libs/xxhash/xxhash.c)

# LibStemmer (stemmer library)
include_directories("${CMAKE_SOURCE_DIR}/libs/libstemmer_c/include")
add_subdirectory("${CMAKE_SOURCE_DIR}/libs/libstemmer_c")

target_link_libraries(blockthetweet "${TORCH_LIBRARIES}" "${SQLITE3_LIB}" stemmer)

set_property(TARGET blockthetweet PROPERTY CXX_STANDARD 20)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  message(STATUS "Torch DLLs found: ${TORCH_DLLS}")
  add_custom_command(TARGET blockthetweet
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:blockthetweet>)
                     
  file(GLOB SQLITE_DLLS "${CMAKE_SOURCE_DIR}/libs/sqlite3/*.dll")
  message(STATUS "SQLite DLLs found: ${SQLITE_DLLS}")
  add_custom_command(TARGET blockthetweet
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${SQLITE_DLLS}
                     $<TARGET_FILE_DIR:blockthetweet>)
endif (MSVC)