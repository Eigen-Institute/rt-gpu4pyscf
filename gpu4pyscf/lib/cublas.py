import ctypes
import os
from cupy.cuda import device
from cupy_backends.cuda.libs import cublas #NOQA

def _load_library(name):
    try:
        return ctypes.CDLL(name)
    except OSError:
        # Try searching in common library paths or nvidia packages
        return None

libcublas = _load_library('libcublas.so')

# Lazy handle initialization
_handle_cache = {}

def get_handle():
    dev_id = device.get_device_id()
    if dev_id not in _handle_cache:
        # Explicitly initialize the handle for this device
        _handle_cache[dev_id] = device.get_cublas_handle()
    return _handle_cache[dev_id]

# Initializing for device 0 at import time can sometimes help "warm up" the cuBLAS context
# for the process, which can avoid CUBLAS_STATUS_NOT_INITIALIZED in some environments.
try:
    get_handle()
except Exception:
    pass

# For backward compatibility if anything uses it directly
def __getattr__(name):
    if name == '_handle':
        return get_handle()
    raise AttributeError(f"module {__name__} has no attribute {name}")