<div align="center">
<img src="https://matteocalafa.com/images/DUBeat-logo.svg" width="500">
</div>
<p align="center">
<a href="https://github.com/teocala/DUBeat"><img src="https://matteocalafa.com/badges/DUBeat-version.svg" /></a>
<a href="https://matteocalafa.com/DUBeat"><img src="https://matteocalafa.com/badges/DUBeat-doc.svg" /></a>
<a href="https://arxiv.org/abs/2406.03045"><img src="https://matteocalafa.com/badges/DUBeat-cite.svg" /></a>
<a href="https://www.gnu.org/licenses/lgpl-3.0.html"><img src="https://matteocalafa.com/badges/DUBeat-license.svg" /></a>
</p>

`DUBeat` is a C++ library that exploits [lifex][] and [deal.II][] to provide discontinuous Galerkin methods on simplices and their applications to cardiac electrophysiology.  
This work originates from a project for the course of *Advanced Programming for Scientific Computing* at *Politecnico di Milano* and it is developed by *Federica Botta* and *Matteo Calafà*.  

The library provides the following two methods:
1. DG with nodal Lagrangian basis.
2. DG with orthogonal Dubiner basis.

Before using the library, read [here][] the installation guidelines and documentation!

In addition, don't forget to cite our [paper][] if you use the library for your research:
```
@article{botta2024highorder,
      title={High-order discontinuous {Galerkin} methods for the monodomain and bidomain models},
      author={Botta, Federica and Calaf{\`a}, Matteo and Africa, Pasquale C. and Vergara, Christian and Antonietti, Paola F.},
      journal={Mathematics in Engineering},
      volume={6},
      number={6},
      pages={726--741},
      year={2024},
      doi={10.3934/mine.2024028}
}
```


[lifex]: https://lifex.gitlab.io/
[here]: https://matteocalafa.com/DUBeat/
[deal.II]: https://www.dealii.org/
[paper]: https://www.aimspress.com/article/doi/10.3934/mine.2024028