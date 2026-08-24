# BCPD

Bayesian Coherent Point Drift implementation in C++23. This repository implements BCPD developed by
https://github.com/ohirose/bcpd in modern c++.

## Setup

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/hulfeldl/BCPD.git
cd BCPD
```

## Build

```bash
# Use KiSync profile for your platform
conan install . --build=missing --profile=cmake/kisync/profiles/linux-clang

# Configure and build
cmake --preset conan-release
cmake --build --preset release
```

## Dependencies

Managed via Conan - see `conanfile.py`.

## CMake & Conan Setup

Provided by [KiSync-cmake](https://github.com/Prometheus-Lorenz/KiSync-cmake) submodule.
