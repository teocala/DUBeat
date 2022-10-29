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

lifex_add_executable(TEST ionic_0d ionic_0d.cpp)
lifex_link_libraries(TEST ionic_0d ${LIB_IONIC})

lifex_create_symlink(TEST ionic_0d "config")

lifex_setup_test(ionic_0d)

foreach(test_name "ttp06_endo" "ttp06_epi" "ttp06_myo")
    lifex_full_target_name(TEST ionic_0d_${test_name} prm_filename)
    lifex_add_test_soft_mpi(
      ionic_0d_${test_name}
      ionic_0d -f ${prm_filename}.prm
               -o output_${test_name})
endforeach()
