A HIGH-ORDER DISCONTINUOUS GALERKIN METHOD AND APPLICATIONS TO CARDIAC ELECTROPHYSIOLOGY
-----------------------------------------------------------------


This sub-library of lifex is dedicated to Discontinuous Galerkin methods and their applications to cardiac electrophysiology.  
This work originates from a project for the course of "*Advanced Programming for Scientific Computing*" and developed by *Federica Botta* and *Matteo Calafà*.  

The implemented and working methods are:
1. Standard DGFEM.
2. DG with Dubiner Basis. 

Both working in either 2 or 3 dimensions and on simplices meshes. FE polynomial orders are limited to the current dealII library availability. As for now, polynomials can be of order at most 2.  

Examples of applications are the problems already supplied in the folder `models`:
1. Poisson's problem.
2. Heat problem.
3. Monodomain problem of cardiac electrophysiology.


### MAIN INSTRUCTIONS FOR THE USE:
- Other applications/problems can be easily implemented. Just follow the above problems structure and override methods at need. Note that the supplement of a new type of DG matrix needs to be added in the `DG_Assemble` methods.
- As for now, simplices meshes can not be built runtime in lifex. We provide at this link a folder with few meshes to be manually copied under `src` folder:  
https://polimi365-my.sharepoint.com/:u:/g/personal/10564633_polimi_it/EW-35JknI6FNmXAOaZrX7JgBI-JHDfoCVZmAMJGODjjdIA?e=4gHxTe  
Otherwise, user-defined meshes (.msh) can be created with any software. In this case, keep in mind to override the method `create_mesh`. 
