Installation and use instructions
-----------------------------------------------------------------
![](./extra/images/title_image.png)
-----------------------------------------------------------------

`DUBeat` is a C++ library that exploits [lifex][] and [deal.II][] to provide discontinuous Galerkin methods on simplices and their applications to cardiac electrophysiology.
This work originates from a project for the course of *Advanced Programming for Scientific Computing* at *Politecnico di Milano* and it is developed by *Federica Botta* and *Matteo Calafà*.

The library provides the following two methods:
1. Standard DGFEM.
2. DG with Dubiner basis.

They both work in either 2 or 3 dimensions depending on the lifex configuration (i.e., specifying `lifex::dim`).


### Dependencies
The library can be used only on a `linux` machine with [CMake][] ≥ `3.22.1` and [GNU bash][] ≥ `5.1.16`.  
Then, `DUBeat 1.0.0` relies almost exclusively on the [lifex][] `1.5.0` installation and its dependencies. Check its [download and install][] page to verify you satisfy all the requirements and install it.  
More precisely, the library has been implemented using the libraries included in the lifex `mk` module, `2022.0` version. This module can be downloaded from [here][] and we refer again to the lifex [download and install][] page for more information. In particular, `DUBeat` uses from this package:
- [ADOL-C][] = `2.7.2`
- [Boost][] = `1.76.0`
- [deal.II][] = `9.3.1`
- [p4est][] = `2.3.2`
- [PETSc][] = `3.15.1`
- [TBB][] = `2021.3.0`
- [Trilinos][] = `13.0.1`

In addition to the core libraries, other packages need to be installed for supplementary reasons:
- [Python][] ≥ `3.9.6` for the creation of convergence plot figures (see generate_convergence_plots.py).
- [Doxygen][] ≥ `1.9.1` for the generation of the library documentation.
- [Graphviz][] ≥ `2.43.0` for the automatic creation of figures included in the documentation.
- [Clang-Format][] ≥ `14.0.0` for the automatic indentation of the library codes
- [gmsh][] ≥ `4.0.4` for the generation of mesh files.
- [ParaView][] ≥ `5.9.1` for the analysis and view of numerical solutions.



### Download and install
1. To download the library, move to the directory where you desire to install `DUBeat` and use the following bash command.
  ```bash
  git clone git@github.com:teocala/DUBeat.git
  ```
  Notice that the previous operation requires the SSH key authentication, see the Github [dedicated page][] for more information.

2. `DUBeat` is an only header library, so what you have just downloaded is already the library installation.

3. Modify the variable `LIFEX_PATH` in the `Makefile.inc` file specifying your local lifex directory.



### Prepare and run your simulation
In the `models` folder we have already provided three different models:
  - Poisson's problem.
  - Heat problem.
  - Monodomain problem of cardiac electrophysiology.

In addition, you find an example of main execution script in the `build` folder.  
To run from this template, you have to follow the five next steps:

1. Choose your model modifying the `main_dubiner_dg.cpp` script.
2. Go back to the main directory and run `make` to compile the script.
3. Now that the execution file is ready, move to the `build` folder and type
  ~~~~
  ./main_dubiner_dg -g
  ~~~~
  to generate the `main_dubiner_dg.prm` parameter file.
4. You can now set your parameters in this file without the need to re-compile. In particular, you can specify the mesh refinement that consists in the choice of the mesh file from the `meshes` folder.
5. Finally, write
  ~~~~
  ./main_dubiner_dg
  ~~~~
  to run your simulation.

### See your results
Results can be viewed and analysed in two ways:
- By default, you should see on the screen the numerical errors with the exact solution. In addition, the `ComputeErrorsDG` class will write the errors on a `.data` file. Finally, you can use the `extra/generate_convergence_plots.py` script to create plots of the errors for different mesh refinements starting from a `.data` file.
- The code generates two solution files, open `solution.xdmf` with [ParaView][] to see the contour plots of the numerical and exact solutions.

### Documentation, indentation and cleaning
The `Make` configuration permits to easily perform some operations that are suggested to keep
a neat and working environment in `DUBeat`, especially for users that want to contribute to
the library or personalize their problems. The following operations are to be
executed on the linux command line from the library main folder:
- Run
  ```bash
  make doc
  ```
  to generate the [Doxygen][] documentation under the `documentation` folder.
- Use instead
  ```bash
  make indent
  ```
  to automatically indent all the files using our customized [Clang-Format][] setup.
- To conclude, you can run
  ```bash
  make clean
  ```
  to remove the previously generated execution files. In addition, you can write
  ```bash
  make distclean
  ```
  to perform a complete cleaning, i.e. removing all generated files such as parameter files, documentation and solution files.


### Personalize your problem
It is very simple to add a new model/problem in `DUBeat`, just follow the next tips!   
- A new model should follow the same structure as the ones in the models folder. This
  implies that the class should derive from `ModelDG` if it is a stationary problem, while it
  should derive from `ModelDG_t` if it is a time-dependent problem.
- The only method that must necessarily be implemented is `assemble_system` that specifies
  how the linear system of the problem is defined. However, also all the other methods can
  be overridden based on the problem choices.
- The addition of a new type of discontinuous Galerkin local matrix needs to be supplemented in the `assemble_DG.hpp` methods.
- As for now, simplices meshes can not be built runtime in [deal.II][]. Therefore, some example meshes are provided in the folder `meshes` and used in the default version.
  In case you need to use your own `.msh` files, you need to use the version of the `create_mesh` method in `ModelDG` that accepts a user-defined mesh path.
- If you need to add new scripts or folders, remember to add them to the `make` and `indent` configurations.



[lifex]: https://lifex.gitlab.io/
[dedicated page]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
[download and install]: https://lifex.gitlab.io/lifex/download-and-install.html
[ParaView]: https://www.paraview.org/
[Python]: https://www.python.org/
[Doxygen]: https://doxygen.nl/
[Graphviz]: https://graphviz.org/
[gmsh]: https://gmsh.info/
[CMake]: https://cmake.org/
[GNU bash]: https://www.gnu.org/software/bash/
[here]: https://github.com/elauksap/mk/releases/download/v2022.0/mk-2022.0-lifex.tar.gz
[Clang-Format]: https://clang.llvm.org/docs/ClangFormat.html
[deal.II]: https://www.dealii.org/
[ADOL-C]: https://github.com/coin-or/ADOL-C
[Boost]: https://www.boost.org/
[p4est]: http://www.p4est.org/
[PETSc]: https://petsc.org/release/
[TBB]: https://github.com/oneapi-src/oneTBB
[Trilinos]: https://trilinos.github.io/
