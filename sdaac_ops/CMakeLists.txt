cmake_minimum_required(VERSION 3.10.2)
project(customsdaacops)

if(DEFINED ENV{SDAA_ROOT} AND EXISTS "$ENV{SDAA_ROOT}/bin/tecocc")
  set(TECOCC_EXECUTABLE "$ENV{SDAA_ROOT}/bin/tecocc")
else()
  find_program(TECOCC_EXECUTABLE NAMES tecocc REQUIRED)
endif()

set(SDAAC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ${SDAA_INC})

file(
  GLOB_RECURSE SDAA_C_OPS
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.scpp)

set(SDAAC_OP_OBJECTS)
foreach(SDAA_C_OP ${SDAA_C_OPS})
  get_filename_component(SDAA_C_OP_NAME ${SDAA_C_OP} NAME_WE)
  set(SDAA_C_OP_OBJECT ${CMAKE_CURRENT_BINARY_DIR}/${SDAA_C_OP_NAME}.o)
  add_custom_command(
    OUTPUT ${SDAA_C_OP_OBJECT}
    COMMAND
      ${TECOCC_EXECUTABLE}
      -I${CMAKE_CURRENT_SOURCE_DIR}
      -I${SDAA_INC}
      -std=c++17
      -O3
      -DNDEBUG
      -flto
      -fPIC
      -c
      ${CMAKE_CURRENT_SOURCE_DIR}/${SDAA_C_OP}
      -o
      ${SDAA_C_OP_OBJECT}
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SDAA_C_OP}
    VERBATIM)
  list(APPEND SDAAC_OP_OBJECTS ${SDAA_C_OP_OBJECT})
endforeach()

set(CUSTOM_SDAA_C_OPS_OUTPUT ${CUSTOM_SDAA_C_OPS_LIB}/libcustomsdaacops.a)
add_custom_command(
  OUTPUT ${CUSTOM_SDAA_C_OPS_OUTPUT}
  COMMAND
    ${TECOCC_EXECUTABLE} ${SDAAC_OP_OBJECTS} -flto -fPIC -shared --sdaa-link
    -sdaa-static-lib -fuse-ld=lld -lm -L ${SDAA_LIB} -o
    ${CUSTOM_SDAA_C_OPS_OUTPUT}
  DEPENDS ${SDAAC_OP_OBJECTS}
  COMMAND_EXPAND_LISTS
  VERBATIM)

add_custom_target(customsdaacops ALL DEPENDS ${CUSTOM_SDAA_C_OPS_OUTPUT})
