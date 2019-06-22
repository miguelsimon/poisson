# Overview

This is an attempt to flesh out one possible approach to spatially localize decomposition events in [NEXT](https://next.ific.uv.es/next/): given the particle counts at a set of detectors in a given time interval, find the voxel where the decomposition event (if any) occurred.

Primary objectives are:
* procrastinate my way out of looking for a flat
* do some math again before I forget all of it
* hone my latex skills until I can bluff and bludgeon myself through any argument just by notational intimidation

An illustration of the approach and a toy solver for the first part of the problem (estimating the model) is found in the inaptly named [poisson_l1.ipynb](poisson_l1.ipynb) notebook, if the github .ipynb renderer feels like working today (it rarely does).

An example of the l1 penalty trick we discussed, described in formula (6) in [Koh, K., Kim, S. J., & Boyd, S. (2007). An interior-point method for large-scale l1-regularized logistic regression. Journal of Machine learning research, 8(Jul), 1519-1555.](http://jmlr.csail.mit.edu/papers/volume8/koh07a/koh07a.pdf) is in [l1_trick.py](l1_trick.py).

# Usage

Fetch dependencies, run tests:
```
make test
```

To make changes I usually run the formatters before the typecheckers (formatting failures are reported as errors):
```
make fmt && make test
```
