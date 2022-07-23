
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
	$(RM) -r ./documentation
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
