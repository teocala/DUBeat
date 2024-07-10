<div align="center">
<img src="https://matteocalafa.com/images/DUBeat-logo.svg" width="500">
</div><br>


<p align="center">
    <a><img src="https://matteocalafa.com/badges/DUBeat-version.svg" /></a>
    <a><img src="https://matteocalafa.com/badges/DUBeat-doc.svg" /></a>
    <a><img src="https://matteocalafa.com/badges/DUBeat-cite.svg" /></a>
    <a><img src="https://matteocalafa.com/badges/DUBeat-license.svg" /></a>
</p>

`DUBeat` is a C++ library that exploits [lifex][] and [deal.II][] to provide discontinuous Galerkin methods on simplices and their applications to cardiac electrophysiology.  
This work originates from a project for the course of *Advanced Programming for Scientific Computing* at *Politecnico di Milano* and it is developed by *Federica Botta* and *Matteo Calafà*.  

The library provides the following two methods:
1. DG with nodal Lagrangian basis.
2. DG with orthogonal Dubiner basis.

Before using the library, read [here][] the installation guidelines and documentation!

In addition, don't forget to cite our [arXiv preprint][] if you use the library for your research:
```
@misc{botta2024highorder,
      title={High-order Discontinuous {Galerkin} Methods for the Monodomain and Bidomain Models}, 
      author={Federica Botta and Matteo Calafà and Pasquale C. Africa and Christian Vergara and Paola F. Antonietti},
      year={2024},
      eprint={2406.03045},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2406.03045}, 
}
```


[lifex]: https://lifex.gitlab.io/
[here]: https://matteocalafa.com/DUBeat/
[deal.II]: https://www.dealii.org/
[arXiv preprint]: https://arxiv.org/abs/2406.03045