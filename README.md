# cnn

[![made-with-python](https://img.shields.io/badge/Made%20with-Python%203.8-1f425f.svg)](https://www.python.org/)
![lint-badge](https://github.com/mkamadeus/cnn/actions/workflows/lint.yml/badge.svg)
![lint-badge](https://github.com/mkamadeus/cnn/actions/workflows/test.yml/badge.svg)

Implementation of Convolutional Neural Networks (CNN) for IF4074 Advanced Machine Learning. Implemented in Python 3.8.

## Notable Libraries

- numpy (for array and matrix operations)
- icecream (for debugging)
- black (code formatting)

## Setup

```bash
python3 -m venv env
source env/bin/activate # in Windows systems, run `.\env\Scripts\activate`
```

## Testing

```bash
make test
```

## Technical Details

- `(n_input, n_channel, w_kernel, h_kernel)` is the shape for the input.
