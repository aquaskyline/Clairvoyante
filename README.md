## Clairvoyante

## Run on MARCC
```
module load tensorflow/r1.0
```
Clairvoyante requires 'intervaltree' library. However, loading the tensorflow module in MARCC changes the default python executable and its library search path. Please use sys.path.append to add the path 'intervaltree' installed at.

