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

enable_testing()

# Tests can be either "soft" or "hard", depending on the time required to run.
add_custom_target(setup_tests DEPENDS symlink_data)

# Wrap "add_custom_target" to execute custom commands
# after building and before running all the tests.
function(lifex_add_setup_test type target)
  if(NOT "${type}" STREQUAL "APP"
      AND NOT "${type}" STREQUAL "EXAMPLE"
      AND NOT "${type}" STREQUAL "TEST")
    message(FATAL_ERROR
      "lifex_add_setup_test: input \"type\" must be either "
      "\"APP\", \"EXAMPLE\" or \"TEST\".")
  endif()

  # Convert ; to spaces for output purposes.
  string(REPLACE ";" " " ARGN_STRING "${ARGN}")

  add_custom_target(
    setup_${LIFEX_PREFIX_${type}}${target}
    COMMAND ./${LIFEX_PREFIX}${LIFEX_PREFIX_${type}}${target} ${ARGN} > /dev/null
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running \"${LIFEX_PREFIX}${LIFEX_PREFIX_${type}}${target} ${ARGN_STRING}\"")
  add_dependencies(setup_${LIFEX_PREFIX_${type}}${target} ${LIFEX_PREFIX_${type}}${target})
  add_dependencies(setup_tests setup_${LIFEX_PREFIX_${type}}${target})
endfunction()

# Wrapper for type == "APP".
function(lifex_setup_app target)
  lifex_add_setup_test(APP ${target} -g -o test)
endfunction()

# Wrapper for type == "EXAMPLE".
function(lifex_setup_example target)
  lifex_add_setup_test(EXAMPLE ${target} -g -o test)
endfunction()

# Wrapper for type == "TEST".
function(lifex_setup_test target)
  lifex_add_setup_test(TEST ${target} -g)
endfunction()


# Wrap "add_test".
function(_lifex_add_test target executable label)
  if(NOT "${label}" STREQUAL "soft"
      AND NOT "${label}" STREQUAL "soft_mpi"
      AND NOT "${label}" STREQUAL "hard"
      AND NOT "${label}" STREQUAL "hard_mpi")
    message(FATAL_ERROR
      "lifex_add_test: input \"label\" must be either "
      "\"soft\", \"soft_mpi\", \"hard\" or \"hard_mpi\".")
  endif()

  if(NOT "${label}" MATCHES "mpi")
    add_test(
      NAME ${LIFEX_PREFIX_TEST}${target}
      COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${LIFEX_PREFIX}${LIFEX_PREFIX_TEST}${executable} ${ARGN}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  else()
    add_test(
      NAME ${LIFEX_PREFIX_TEST}${target}
      COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${MPIEXEC_MAX_NUMPROCS}
              ${MPIEXEC_PREFLAGS} ${CMAKE_CURRENT_BINARY_DIR}/${LIFEX_PREFIX}${LIFEX_PREFIX_TEST}${executable}
              ${MPIEXEC_POSTFLAGS} ${ARGN}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  endif()

  string(REPLACE "_mpi" "" label_filtered "${label}")
  set_tests_properties(
    ${LIFEX_PREFIX_TEST}${target}
    PROPERTIES LABELS "${LIFEX_PREFIX_TEST}${label_filtered}")
  add_dependencies(setup_tests ${LIFEX_PREFIX_TEST}${executable})
endfunction()

# Export aliases for soft tests.
function(lifex_add_test_soft target executable)
  _lifex_add_test(${target} ${executable} soft ${ARGN})
endfunction()

function(lifex_add_test_soft_mpi target executable)
  _lifex_add_test(${target} ${executable} soft_mpi ${ARGN})
endfunction()

# Export aliases for hard tests.
function(lifex_add_test_hard target executable)
  _lifex_add_test(${target} ${executable} hard ${ARGN})
endfunction()

function(lifex_add_test_hard_mpi target executable)
  _lifex_add_test(${target} ${executable} hard_mpi ${ARGN})
endfunction()


# Add tests for restart: two tests are performed,
# with suffixes restart_pre and restart_post, respectively.
# The two tests are configured so that restart_post runs after restart_pre.
#They are meant to be used to check that restart functionalities works.
function(_lifex_add_test_restart target executable label)
  if(NOT "${label}" STREQUAL "soft"
      AND NOT "${label}" STREQUAL "soft_mpi"
      AND NOT "${label}" STREQUAL "hard"
      AND NOT "${label}" STREQUAL "hard_mpi")
    message(FATAL_ERROR
      "lifex_add_test: input \"label\" must be either "
      "\"soft\", \"soft_mpi\", \"hard\" or \"hard_mpi\".")
  endif()

  lifex_full_target_name(TEST ${target}_restart_pre prm_filename)
  _lifex_add_test(
    ${target}_restart_pre
    ${executable} ${label}
    -f ${prm_filename}.prm -o output_restart_pre)

  lifex_full_target_name(TEST ${target}_restart_post prm_filename)
  _lifex_add_test(
    ${target}_restart_post
    ${executable} ${label}
    -f ${prm_filename}.prm -o output_restart_post)

  # Make sure that the restart_post test is run after the restart_pre one.
  # Skip restart_post if restart_pre fails.
  set_tests_properties(
    ${LIFEX_PREFIX_TEST}${target}_restart_pre
    PROPERTIES FIXTURES_SETUP ${LIFEX_PREFIX_TEST}${target}_restart)
  set_tests_properties(
    ${LIFEX_PREFIX_TEST}${target}_restart_post
    PROPERTIES FIXTURES_REQUIRED ${LIFEX_PREFIX_TEST}${target}_restart)
endfunction()

# Export aliases for soft restart tests.
function(lifex_add_test_restart_soft target executable)
  _lifex_add_test_restart(${target} ${executable} soft ${ARGN})
endfunction()

function(lifex_add_test_restart_soft_mpi target executable)
  _lifex_add_test_restart(${target} ${executable} soft_mpi ${ARGN})
endfunction()

# Export aliases for hard restart tests.
function(lifex_add_test_restart_hard target executable)
  _lifex_add_test_restart(${target} ${executable} hard ${ARGN})
endfunction()

function(lifex_add_test_restart_hard_mpi target executable)
  _lifex_add_test_restart(${target} ${executable} hard_mpi ${ARGN})
endfunction()
