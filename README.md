![](./dinosaur-logo.png)

# Dinosaur: differentiable dynamics for global atmospheric modeling

Dinosaur is a spectral dynamical core for global atmospheric modeling written in
JAX:

- *Dynamics*: Dinosaur solves the shallow water equations, and the primitive equations (moist and dry) on sigma coordinates.
- *Auto-diff*: Dinosaur supports both forward- and backward-mode automatic differentiation in JAX.
- *Acceleration*: Dinosaur is designed to run efficiently on modern accelerator
hardware (GPU/TPU), including parallelization across multiple devices.

For more details, see our paper on [Neural General Circulation Models](https://arxiv.org/abs/2311.07222).

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for details. We are open to user
contributions, but please reach out (either on GitHub or by email) to coordinate
before starting significant work.

## License

Apache 2.0; see [`LICENSE`](LICENSE) for details.
