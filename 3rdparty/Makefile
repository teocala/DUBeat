## ---------------------------------------------------------------------
## Copyright (C) 2022 by the DUBeat authors.
##
## This file is part of DUBeat.
##
## DUBeat is free software; you can redistribute it and/or modify
## it under the terms of the GNU Lesser General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## DUBeat is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
## Lesser General Public License for more details.
##
## You should have received a copy of the GNU Lesser General Public License
## along with DUBeat.  If not, see <http://www.gnu.org/licenses/>.
## ---------------------------------------------------------------------

# Author: Matteo Calafà <matteo.calafa@mail.polimi.it>.

##
## This script has been readapted from the corresponding file
## available at the pacs-examples repository
## (https://github.com/pacs-course/pacs-examples),
## released under compatible license terms.
##

#
# The make file used for the DUBeat 3rdparty libraries.
#


-include ../Makefile.inc

#==========================
.PHONY = all lifex0d

.DEFAULT_GOAL = all


all: lifex0d

lifex0d: 
	@cd lifex-dev_0d; mkdir -p build; cd build; cmake ..
	@cd lifex-dev_0d/build; make 