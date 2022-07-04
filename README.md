A HIGH-ORDER DISCONTINUOUS GALERKIN METHOD AND APPLICATIONS TO CARDIAC ELECTROPHYSIOLOGY
-----------------------------------------------------------------


This sub-library of LifeX is dedicated to discontinuous Galerkin methods and their applications to cardiac electrophysiology.  
This work originates from a project for the course of "*Advanced Programming for Scientific Computing*" and developed by *Federica Botta* and *Matteo Calafà*.  

The implemented and working methods are:
1. Standard DGFEM.
2. DG with Dubiner Basis. 

Both working in either 2 or 3 dimensions and on simplices meshes. FE polynomial orders are limited to the current deal.II library availability. As for now, polynomials can be of order at most 2.  

Examples of applications are the problems already supplied in the folder `models`:
1. Poisson's problem.
2. Heat problem.
3. Monodomain problem of cardiac electrophysiology.


### MAIN INSTRUCTIONS FOR THE USE:
- Other applications/problems can be easily implemented. Just follow the above problems structure and overridden methods at need. Note that the supplement of a new type of DG matrix needs to be added in the `DG_Assemble` methods. 
- As for now, simplices meshes can not be built runtime in LifeX. Therefore, some reference meshes are provided in the folder `meshes`. In case you need to use your own mesh files, you need to change the default create_mesh function.
