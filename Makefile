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
# The make file used in the DUBeat library.
#


-include Makefile.inc
# build directory
DIR =./build
# get all files *.cpp
SRCS=$(wildcard $(DIR)/*.cpp)
# get the corresponding object file
OBJS = $(SRCS:.cpp=.o)
# get all headers in the working directory
HEADERS=$(wildcard *.hpp)
#
exe_sources=$(filter $(DIR)/main%.cpp,$(SRCS))
EXEC=$(exe_sources:.cpp=)

DOXYFILE = ./Doxyfile

#==========================
.PHONY = all doc clean distclean

.DEFAULT_GOAL = all

all: $(DEPEND) $(EXEC)

clean:
	$(RM) -f $(EXEC) $(OBJS)
	cd $(DIR); $(RM) *.out *.bak *~ *.aux *.log

distclean:
	$(MAKE) clean
	$(RM) -r ./documentation/html ./documentation/latex
	cd $(DIR); $(RM) -f *.h5 *.xdmf *.prm *.data

doc:
	doxygen $(DOXYFILE)

$(EXEC): $(OBJS)

$(OBJS): $(SRCS)

$(DEPEND): $(SRCS)
	-\rm $(DEPEND)
	for f in $(SRCS); do \
	$(CXX) $(STDFLAGS) $(CPPFLAGS) -MM $$f >> $(DEPEND); \
	done

-include $(DEPEND)
