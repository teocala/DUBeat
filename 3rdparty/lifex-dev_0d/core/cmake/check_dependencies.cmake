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

# Add include directories.
include_directories(${CMAKE_SOURCE_DIR}/)

if(NOT LIFEX_CORE_STANDALONE)
  include_directories(${CMAKE_SOURCE_DIR}/core)
endif()

set(CMAKE_CXX_STANDARD "17")
set(CMAKE_CXX_STANDARD_REQUIRED "ON")

# Fix warning on macOS.
if("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
  set(CMAKE_MACOSX_RPATH "ON")
endif()


# Set default build type to Release.
if(NOT CMAKE_BUILD_TYPE OR "${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()
message(STATUS)
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS)
if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
  add_definitions(-DBUILD_TYPE_DEBUG)
endif()

# Locate MPI compiler.
find_package(MPI REQUIRED)
set(CMAKE_CXX_COMPILER "${MPI_CXX_COMPILER}")


# Locate Boost.
find_package(Boost 1.72.0 REQUIRED
  COMPONENTS filesystem iostreams serialization
  HINTS ${BOOST_DIR} $ENV{BOOST_DIR} $ENV{mkBoostPrefix})
message(STATUS "Using the Boost-${Boost_VERSION} configuration found at ${Boost_DIR}")
message(STATUS)
include_directories(${Boost_INCLUDE_DIRS})


# Locate deal.II and initialize its variables.
find_package(deal.II 9.3.1 REQUIRED
  HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR} $ENV{mkDealiiPrefix})
deal_ii_initialize_cached_variables()


# Determine linear algebra backend.
set(LIN_ALG "Trilinos" CACHE STRING "Use Trilinos or PETSc as deal.II linear algebra backend.")

message(STATUS)
if("${LIN_ALG}" STREQUAL "Trilinos")
  add_definitions(-DLIN_ALG_TRILINOS)
elseif("${LIN_ALG}" STREQUAL "PETSc")
  add_definitions(-DLIN_ALG_PETSC)
else()
  message(FATAL_ERROR "Please select deal.II linear algebra backend with -DLinAlg=Trilinos or -DLinAlg=PETSc.")
endif()
message(STATUS "Using ${LIN_ALG} as deal.II linear algebra backend.")


# Locate VTK.
find_package(VTK 9.0.0 REQUIRED
  HINTS ${VTK_DIR} $ENV{VTK_DIR} $ENV{mkVtkPrefix})
unset(VTK_MPI_NUMPROCS CACHE) # Remove unused variable from cache.
message(STATUS "Using the VTK-${VTK_VERSION} installation found at ${VTK_PREFIX_PATH}")


# Enable/disable interface methods with pyfex.
set(WITH_PYFEX OFF CACHE BOOL "Enable/disable interface with pyfex.")

if(${WITH_PYFEX})
  message(STATUS)
  add_definitions(-DWITH_PYFEX)
  message(STATUS "pyfex interface: enabled.")
  message(STATUS)
endif()


# Add useful compiler flags.
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wfloat-conversion -Wmissing-braces -Wnon-virtual-dtor")


# Add option and flags for sanitizer tools.
set(CMAKE_ENABLE_SANITIZERS ON CACHE BOOL "Enable/disable compiler-generated sanitizer runtime instrumentation.")

if("${CMAKE_ENABLE_SANITIZERS}" AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    message(STATUS)
    if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(SANITIZER_FLAGS "-fsanitize=undefined -fno-sanitize=vptr -fno-sanitize-recover")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(SANITIZER_FLAGS "-check=conversions -check=stack -check-pointers=rw -check-pointers-dangling=all -check=uninit")
    else()
        message(WARNING "Sanitizer flags not enabled for compiler ${CMAKE_CXX_COMPILER_ID}")
        set(SANITIZER_FLAGS "")
    endif()
    message(STATUS "Sanitizer flags: ${SANITIZER_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZER_FLAGS}")
endif()


# Add option and flags for coverage.
set(CMAKE_ENABLE_COVERAGE OFF CACHE BOOL "Enable/disable coverage flags")

if("${CMAKE_ENABLE_COVERAGE}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} --coverage")

  message(STATUS)
  check_cxx_compiler_flag(-fprofile-abs-path HAVE_fprofile_abs_path)
  if(HAVE_fprofile_abs_path)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-abs-path")
  endif()
endif()
message(STATUS)
