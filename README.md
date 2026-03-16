Finite volume discretization.

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

## Description

Discretization with finite volume of some equation.

TODO: description of what you do in some details, with link to relevant docs.

## Contents

The repository is organized as follows:

`demo/` Demonstration codes, which rely on the `finite_volume` package.

`src/dirac_operator` Provides the `finite_volume` package. 

## Development

### Local installation using anaconda

The package and its dependencies can be installed in a new conda environment with:

```console
    # Create and activate development environment
conda env create --file environment.yml 
conda activate finite_volume
    # Install package and development dependencies
pip install -e .[dev]
```

To use the environment in VS code, open the command palette: 

- `File: Open Folder` and select root of this repository.
- `Python: Select Interpreter` and select the `finite_volume` conda environment.

## Background on how the repository is setup

Both the package and the repository have been set up by following some of the guidelines from [scientific-python](https://github.com/scientific-python/cookie).

- Metadata are defined using `pyproject.toml` following the [official specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/).
- The build backend is [`hatchling`](https://hatch.pypa.io/latest/), which is well-suited to pure python packages and follows the official specification.
- A `.gitignore` file is provided to avoid committing unwanted files.