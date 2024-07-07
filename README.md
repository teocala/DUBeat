-----------------------------------------------------------------
![](./extra/images/title_image.png)
-----------------------------------------------------------------
<p align="left">
    <a><img src="https://img.shields.io/badge/Version%20-%201.0.1%20-%20%20%230000FF" /></a>
    <a href="https://matteocalafa.com/DUBeat/"><img src="https://img.shields.io/badge/Documentation%20-%20matteocalafa.com%2FDUBeat%20-%20%231E90FF" /></a>
    <a href="https://arxiv.org/abs/2406.03045"> <img src="https://img.shields.io/badge/Cite%20-%20arXiv%3A2406.03045%20-%20%2332CD32" /></a>
</p>

`DUBeat` is a C++ library that exploits [lifex][] and [deal.II][] to provide discontinuous Galerkin methods on simplices and their applications to cardiac electrophysiology.  
This work originates from a project for the course of *Advanced Programming for Scientific Computing* at *Politecnico di Milano* and it is developed by *Federica Botta* and *Matteo Calafà*.  

The library provides the following two methods:
1. DG with nodal Lagrangian basis.
2. DG with orthogonal Dubiner basis.

Before using the library, read [here][] the installation guidelines and documentation!

In addition, don't forget to cite our arXiv preprint if you use the library for your research:
```
@misc{botta2024highorderdiscontinuousgalerkinmethods,
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
