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

#
# The make file used in the DUBeat library.
#

LIFEX_PATH?=/home/matteo/lifex-dev

lifexInc=$(LIFEX_PATH)/core
lifexLib=$(LIFEX_PATH)/build/lib
mkDealiiInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/include
mkDealiiLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/dealii/9.3.1/lib
mkAdolcInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/include
mkAdolcLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/adol-c/2.7.2/lib64
mkTrilinosInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/include
mkTrilinosLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/trilinos/13.0.1/lib
mkBoostInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/include
mkBoostLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/boost/1.76.0/lib
mkTbbInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/tbb/2021.3.0/include
mkTbbLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/tbb/2021.3.0/lib
mkP4estInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.3.2/include
mkP4estLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/p4est/2.3.2/lib
mkPetscInc?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.15.1/include
mkPetscLib?=/u/sw/toolchains/gcc-glibc/11.2.0/pkgs/petsc/3.15.1/lib



CPPFLAGS+= -DLIN_ALG_TRILINOS -I$(mkDealiiInc) -I$(lifexInc) -I$(mkAdolcInc) -I$(mkTrilinosInc) -I$(mkBoostInc) -I$(mkTbbInc) -I$(mkP4estInc) -I$(mkPetscInc)
LDLIBS+=-L$(mkDealiiLib) -ldeal_II -L$(lifexLib) -Wl,-rpath=$(lifexLib) -llifex_core -L$(mkAdolcLib) -L$(mkTrilinosLib) -lepetra -lteuchoscore -lteuchoscomm -L$(mkBoostLib) -L$(mkTbbLib) -ltbb -L$(mkP4estLib) -L$(mkPetscLib) -lstdc++ -ldl -lm
