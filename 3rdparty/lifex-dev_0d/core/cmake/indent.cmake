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

# Provide an indentation target for indenting uncommitted changes
# and changes to the current branch.
add_custom_target(indent
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./scripts/format/indent
  COMMENT "Indenting recently changed files in the lifex directories")

# Provide an indentation target for indenting all header and source files.
add_custom_target(indent_all
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  COMMAND ./scripts/format/indent_all
  COMMENT "Indenting all files in the lifex directories")
