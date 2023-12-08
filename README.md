# Dinosaur: A differentiable dynamical core for atmospheric GCMs.

Dinosaur is a package for numerical simulation of atmospheric dynamics. It is
implemented in JAX and hence supports automatic differentiation. It uses
pseudo-spectral discretization for solving the primitive equations and is
optimized to run well on modern accelerators (TPUs and GPUs).

For details see [Neural General Circulation Models](https://arxiv.org/abs/2311.07222)
publication (arxiv 2023).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.

## Disclaimer

This project is not an official Google project. It is not supported by
Google and Google specifically disclaims all warranties as to its quality,
merchantability, or fitness for a particular purpose.
