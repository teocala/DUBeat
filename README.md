-----------------------------------------------------------------
![](./extra/images/title_image.png)
-----------------------------------------------------------------

`DUBeat` is a C++ library that exploits [lifex][] and [deal.II][] to provide discontinuous Galerkin methods on simplices and their applications to cardiac electrophysiology.  
This work originates from a project for the course of *Advanced Programming for Scientific Computing* at *Politecnico di Milano* and it is developed by *Federica Botta* and *Matteo Calafà*.  

The implemented and working methods are:
1. Standard DGFEM.
2. DG with Dubiner Basis.

They both work in either 2 or 3 dimensions depending on the lifex configuration (specifying `lifex::dim`). Instead, the discretization polynomial order is restricted to the current [deal.II][] availability and, as for now, it can be at most 2.



### Dependencies
`DUBeat 1.0.0` relies almost exclusively on the [lifex][] `1.4.0` installation and its dependencies. Check its [download and install][] page to verify you satisfy all its requirements.   
In particular, the library has been implemented using:
- [CMake][] = `3.16.3`
- [Doxygen][] = `1.8.17`
- [Graphviz][] = `2.42.2`  

and other libraries included in the lifex `mk` module, `2022.0` version. This module can be downloaded from [here][] and we refer again to the [download and install][] page for its installation. In particular, this package contains:
- [ADOL-C][] = `2.7.2`
- [Boost][] = `1.76.0`
- [deal.II][] = `9.3.1`
- [p4est][] = `2.3.2`
- [PETSc][] = `3.15.1`
- [TBB][] = `2021.3.0`
- [Trilinos][] = `13.0.1`



### Download and install
1. To download the library, move to the directory where you desire to install `DUBeat` and use the following bash command.
  ```bash
  git clone git@github.com:teocala/PACS_Project.git
  ```
  Notice that the previous operation requires the SSH key authentication, see the Github [dedicated page][] for more information.

2. `DUBeat` is an only header library, so what you have just downloaded is already the library installation.

3. Modify the variable `LIFEX_PATH` in the `Makefile.inc` file specifying your local lifex directory.  

### Prepare your simulation
In the `models` folder we have already provided three different models:
  - Poisson's problem.
  - Heat problem.
  - Monodomain problem of cardiac electrophysiology.    

In addition, you find an example main execution script in the `build` folder.   

To obtain your executable, you then need to
1. Choose your model modifying the `main_dubiner_dg.cpp` script.
2. Go back to the main directory and run `make` to compile the script.

### Run your simulation
1. Now that the execution file is ready, move to the `build` folder and type
    ```bash
    ./main_dubiner_dg -g
    ```
    to generate the `main_dubiner_dg.prm` parameter file.   
2. You can now set your parameters in this file without the need to re-compile. In particular, you can specify the mesh refinement that consists in the choice of the mesh file from the `meshes` folder.   
3. Write
    ```bash
    ./main_dubiner_dg
    ```
    to run your simulation.

### See your results
Results can be viewed in two ways:
- By default, you should see on the screen the numerical errors with the exact solution. In addition, the `DG_error_parser` class will write the errors on a `.data` file. Finally, you can use the `generate_convergence_plots.py` script to create plots of the errors for different mesh refinements starting from a `.data` file.
- The code generates two solution files, open `solution.xdmf` with [ParaView][] to see the contour plots of the numerical and exact solutions.

### Documentation, indentation and cleaning
- In the main folder, run
  ```bash
  make doc
  ```
  to generate the [Doxygen][] documentation under the `documentation` folder.
- Still in the main folder, run
  ```bash
  ./extra/indent/indent_all
  ```
  to automatically indent all the files using our customized [Clang-Format] setup.
- Once again in the main folder, use
  ```bash
  make clean
  ```
  to remove the previously generated execution files. In addition, you can write
  ```bash
  make distclean
  ```  
  to perform a complete cleaning, i.e. removing all generated files such as parameter files, documentation and solution files.

### Personalize your problem
- Other applications/problems can be easily implemented. Just implement your own model header as in the `models` folder!  
  Notice that the addition of a new type of DG matrix needs to be supplemented in the `DG_Assemble.hpp` methods.
- As for now, simplices meshes can not be built runtime in [deal.II][]. Therefore, some example meshes are provided in the folder `meshes` and used in the default version.
  In case you need to use your own `.msh` files, you need to use the version of the `create_mesh` method in `Model_DG` that accepts a user mesh path.
- If you need to add new scripts/folders, remember to add them to the `make` and `indent` configurations.



[lifex]: https://lifex.gitlab.io/
[dedicated page]: https://docs.github.com/en/authentication/connecting-to-github-with-ssh
[download and install]: https://lifex.gitlab.io/lifex/download-and-install.html
[ParaView]: https://www.paraview.org/
[Doxygen]: https://doxygen.nl/
[Graphviz]: https://graphviz.org/
[CMake]: https://cmake.org/
[here]: https://github.com/elauksap/mk/releases/download/v2022.0/mk-2022.0-lifex.tar.gz
[Clang-Format]: https://clang.llvm.org/docs/ClangFormat.html
[deal.II]: https://www.dealii.org/
[ADOL-C]: https://github.com/coin-or/ADOL-C
[Boost]: https://www.boost.org/
[p4est]: http://www.p4est.org/
[PETSc]: https://petsc.org/release/
[TBB]: https://github.com/oneapi-src/oneTBB
[Trilinos]: https://trilinos.github.io/
