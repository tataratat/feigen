# feigen - python interactive Front End for IGA ENgines.
feigen is a python library for interactive spline plotting.
It focues on supporting isogeometric analysis tools, such as `iganet`, `splinepy`, and `pygismo`.

## Install guide
you can install feigen using `pip`:
```
pip install feigen
```
For the latest develop version:
```
pip install git+https://github.com/tataratat/feigen.git@main
```

## Quick start
### iganet
Current version supports iganet's BSplineSurface.
Assuming that you have a server running,
```
python3 -c "import feigen; feigen.BSpline2D('ws://localhost:9001').start()"
```

### IGA examples
#### Poisson problem 2D
```
python3 -c "import feigen; feigen.Poisson2D().start()"
```

#### Poisson problem 2D - Configurable and with option to view collocations points
```
python3 -c "import feigen; feigen.CustomPoisson2D().start()"
```

### Spline Examples
#### Jacobian Determinant 2D
```
python3 -c "import feigen; feigen.JacobianDeterminant().start()"
```

### NURBS Weights 2D
```
python3 -c "import feigen; feigen.NURBSWeights().start()"
```
