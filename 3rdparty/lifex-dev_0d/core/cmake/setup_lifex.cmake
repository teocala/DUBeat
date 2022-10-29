## ---------------------------------------------------------------------
## Copyright (C) 2019 - 2022 by the lifex authors.
##
## This file is part of lifex.
##
## lifex is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## lifex is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with lifex.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

# Author: Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.

# Set output folder.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")


# Install license files.
install(
  FILES COPYRIGHT.md LICENSE.md
  DESTINATION ${CMAKE_INSTALL_PREFIX})

install(
  DIRECTORY doc/licenses
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc)

# Install logos.
install(
  DIRECTORY doc/logo
  DESTINATION ${CMAKE_INSTALL_PREFIX}/share/doc)

# Install lifexrun executable.
install(
  PROGRAMS apps/lifexrun
  DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)


# Define target to link data directories.
add_custom_target(symlink_data ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/data
    ${CMAKE_BINARY_DIR}/data
  COMMAND ${CMAKE_COMMAND} -E create_symlink
    ${CMAKE_SOURCE_DIR}/mesh
    ${CMAKE_BINARY_DIR}/mesh
  COMMENT "Creating symbolic links to data and mesh directories")

# Define "lifex_set" to define variables such as
# library names. CACHE INTERNAL makes the variable
# visible globally.
function(lifex_set name value)
  set(${name} "${value}" CACHE INTERNAL "")
endfunction()

# Return "${prefix}_${type}_${target}" given an executable type.
function(lifex_full_target_name type target output)
  lifex_target_name(${type} ${target} target_name)

  set(${output} "${LIFEX_PREFIX}${target_name}" PARENT_SCOPE)
endfunction()

# Return "${type}_${target}" given an executable type.
function(lifex_target_name type target output)
  if(NOT "${type}" STREQUAL "APP"
      AND NOT "${type}" STREQUAL "EXAMPLE"
      AND NOT "${type}" STREQUAL "TEST")
    message(FATAL_ERROR
      "lifex_target_name: input \"type\" must be either "
      "\"APP\", \"EXAMPLE\" or \"TEST\".")
  endif()

  set(${output} "${LIFEX_PREFIX_${type}}${target}" PARENT_SCOPE)
endfunction()

# Define "lifex_create_symlink" to link custom data files
# to the build directory.
# The command for creating the symbolic link is added
# as a dependency of a given input target.
function(lifex_create_symlink type target filename)
  if(NOT "${type}" STREQUAL "APP"
      AND NOT "${type}" STREQUAL "EXAMPLE"
      AND NOT "${type}" STREQUAL "TEST")
    message(FATAL_ERROR
      "lifex_create_symlink: input \"type\" must be either "
      "\"APP\", \"EXAMPLE\" or \"TEST\".")
  endif()

  lifex_target_name(${type} ${target} target)

  add_custom_target(symlink_${filename}_${target}
    COMMAND ${CMAKE_COMMAND} -E create_symlink
      ${CMAKE_CURRENT_SOURCE_DIR}/${filename}
      ${CMAKE_CURRENT_BINARY_DIR}/${filename}
    COMMENT "Creating symbolic link to ${filename} for target ${target}")
  add_dependencies(${target} symlink_${filename}_${target})
endfunction()


# Re-define "add_library" and "add_executable"
# to add target prefix and to install them.
function(lifex_add_library target)
  add_library(${target} SHARED ${ARGN})
  set_target_properties(${target} PROPERTIES
                        PREFIX "lib${LIFEX_PREFIX}"
                        VERSION ${LIFEX_VERSION})
  install(TARGETS ${target} LIBRARY DESTINATION lib)
endfunction()

function(lifex_add_executable type target)
  if(NOT "${type}" STREQUAL "APP"
      AND NOT "${type}" STREQUAL "EXAMPLE"
      AND NOT "${type}" STREQUAL "TEST")
    message(FATAL_ERROR
      "lifex_add_executable: input \"type\" must be either "
      "\"APP\", \"EXAMPLE\" or \"TEST\".")
  endif()

  lifex_target_name(${type} ${target} target)

  add_executable(${target} ${ARGN})
  set_target_properties(${target} PROPERTIES PREFIX "${LIFEX_PREFIX}")

  # Only install "APP" executables.
  if("${type}" STREQUAL "APP")
    install(TARGETS ${target} RUNTIME DESTINATION bin)
  endif()
endfunction()


# Re-define "target_link_libraries" to
# add dependency on target "symlink_data" and to
# automatically call "deal_ii_setup_target".
function(lifex_link_libraries type target)
  if(NOT "${type}" STREQUAL "APP"
      AND NOT "${type}" STREQUAL "EXAMPLE"
      AND NOT "${type}" STREQUAL "LIB"
      AND NOT "${type}" STREQUAL "TEST")
    message(FATAL_ERROR
      "lifex_link_libraries: input \"type\" must be either "
      "\"APP\", \"EXAMPLE\", \"LIB\" or \"TEST\".")
  endif()

  if(NOT "${type}" STREQUAL "LIB")
    lifex_target_name(${type} ${target} target)
  endif()

  if("${target}" STREQUAL ${LIB_CORE})
    target_link_libraries(${target} ${ARGN})
  else()
    target_link_libraries(${target} ${LIB_CORE} ${ARGN})
  endif()
  add_dependencies(${target} symlink_data)
  deal_ii_setup_target(${target})
endfunction()

# Add all subdirectories containing a CMakeLists.txt file, in alphabetical order.
function(add_all_subdirectories)
  file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

  foreach(child ${children})
    if(IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/${child}
        AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${child}/CMakeLists.txt)
      add_subdirectory(${child})
    endif()
  endforeach()
endfunction()

# Find source files (*.cpp) in the current directory.
function(lifex_find_sources output)
  file(GLOB srcs CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
  set(${output} "${srcs}" CACHE INTERNAL "")
endfunction()

# Find source files (*.cpp) recursively.
# A list of relative paths can be optionally provided.
function(lifex_find_sources_recursively output)
  set(paths "${ARGN}")

  if(paths)
    list(TRANSFORM paths APPEND "/*.cpp")
  else()
    set(paths "*.cpp")
  endif()

  file(GLOB_RECURSE srcs CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${paths})
  set(${output} "${srcs}" CACHE INTERNAL "")
endfunction()
