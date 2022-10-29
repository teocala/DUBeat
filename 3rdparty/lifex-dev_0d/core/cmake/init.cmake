## ---------------------------------------------------------------------
## Copyright (C) 2022 by the lifex authors.
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

# Initialize and set common CMake variables.
set(LIFEX_NAME lifex)

# Get lifex version number.
if(LIFEX_CORE_STANDALONE)
  file(STRINGS "${CMAKE_SOURCE_DIR}/VERSION_lifex" LIFEX_VERSION LIMIT_COUNT 1)
else()
  file(STRINGS "${CMAKE_SOURCE_DIR}/core/VERSION_lifex" LIFEX_VERSION LIMIT_COUNT 1)
endif()

string(REGEX REPLACE "^([0-9]+)\\..*" "\\1"                LIFEX_VERSION_MAJOR "${LIFEX_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1"          LIFEX_VERSION_MINOR "${LIFEX_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" LIFEX_VERSION_PATCH "${LIFEX_VERSION}")

#
message(STATUS "********************************")
message(STATUS "*** Configuring ${LIFEX_NAME} v${LIFEX_VERSION} ***")
message(STATUS "********************************")
message(STATUS)

set(LIFEX_PREFIX         "${LIFEX_NAME}_")
set(LIFEX_PREFIX_APP     "")
set(LIFEX_PREFIX_EXAMPLE "example_")
set(LIFEX_PREFIX_TEST    "test_")
