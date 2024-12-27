import os

BACKEND = os.environ.get("PYGP_BACKEND", "G")

if BACKEND == "G":
    from . import ndarray as array_api
    NDArray = array_api.NDArray
else:
    import np as array_api
    NDArray = array_api.ndarray
    
    
    