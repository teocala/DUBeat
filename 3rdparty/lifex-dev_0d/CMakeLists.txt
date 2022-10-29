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

cmake_minimum_required(VERSION 3.12.0)

include(${CMAKE_SOURCE_DIR}/core/cmake/init.cmake)

project(${LIFEX_NAME} VERSION ${LIFEX_VERSION} LANGUAGES CXX C)

include(${CMAKE_SOURCE_DIR}/core/cmake/check_dependencies.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/core/cmake/indent.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/core/cmake/setup_lifex.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/core/cmake/setup_lifex_tests.cmake)

# Generate core.
add_subdirectory(${CMAKE_SOURCE_DIR}/core/)

# Generate physical models.
add_subdirectory(${CMAKE_SOURCE_DIR}/source/)

# Generate tests.
add_subdirectory(${CMAKE_SOURCE_DIR}/tests/)
