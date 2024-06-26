## ---------------------------------------------------------------------
## Copyright (C) 2024 by the DUBeat authors.
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
# doxyfile name
DOXYFILE = ./Doxyfile
# indent command directory
INDENT = ./extra/indent

#==========================
.PHONY = all doc clean distclean

.DEFAULT_GOAL = all


all: check_lifex $(DEPEND) $(EXEC) end_print

check_lifex:
	@if [ ! -d $(LIFEX_PATH) ]; then echo "\033[91m\nLIFEX_PATH is not correct, set your local lifex installation path in Makefile.inc\n\033[0m"; fi

end_print:
	@echo "\nDUBeat version 1.0.1: compilation completed\n"

clean:
	@$(RM) -f $(EXEC) $(OBJS)
	@cd $(DIR); $(RM) *.out *.bak *~ *.aux *.log

distclean:
	@$(MAKE) -s clean
	@$(RM) -r ./documentation/html ./documentation/latex
	@cd $(DIR); $(RM) -f *.h5 *.xdmf *.prm *.data

doc:
	doxygen $(DOXYFILE)

indent:
	@$(INDENT)/indent_all
	@echo "DUBeat version 1.0.1: indentation completed."

$(EXEC): $(OBJS) 

$(OBJS): $(SRCS)

$(DEPEND): $(SRCS)
	-\rm $(DEPEND)
	for f in $(SRCS); do \
	$(CXX) $(STDFLAGS) $(CPPFLAGS) -MM $$f >> $(DEPEND); \
	done

-include $(DEPEND)
