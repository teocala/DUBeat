LIFEX_PATH?=/home/matteo/lifex

lifexInc=$(LIFEX_PATH)
lifexLib=$(LIFEX_PATH)/build/lib
mkDealiiInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/dealii/9.3.1/include
mkDealiiLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/dealii/9.3.1/lib
mkAdolcInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/adol-c/2.7.2/include
mkAdolcLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/adol-c/2.7.2/lib64
mkTrilinosInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/trilinos/13.0.1/include
mkTrilinosLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/trilinos/13.0.1/lib
mkBoostInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/boost/1.76.0/include
mkBoostLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/boost/1.76.0/lib
mkTbbInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/tbb/2021.3.0/include
mkTbbLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/tbb/2021.3.0/lib
mkP4estInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/p4est/2.3.2/include
mkP4estLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/p4est/2.3.2/lib
mkPetscInc?=/u/sw/toolchains/gcc-glibc/11/pkgs/petsc/3.15.1/include
mkPetscLib?=/u/sw/toolchains/gcc-glibc/11/pkgs/petsc/3.15.1/lib



CPPFLAGS+= -DLIN_ALG_TRILINOS -I$(mkDealiiInc) -I$(lifexInc) -I$(mkAdolcInc) -I$(mkTrilinosInc) -I$(mkBoostInc) -I$(mkTbbInc) -I$(mkP4estInc) -I$(mkPetscInc)
LDLIBS+=-L$(mkDealiiLib) -ldeal_II -L$(lifexLib) -Wl,-rpath=$(lifexLib) -llifex_core -L$(mkAdolcLib) -L$(mkTrilinosLib) -lepetra -lteuchoscore -lteuchoscomm -L$(mkBoostLib) -L$(mkTbbLib) -ltbb -L$(mkP4estLib) -L$(mkPetscLib) -lstdc++ -ldl -lm