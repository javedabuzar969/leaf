import numpy as np

# Mock top-level TensorFlow functions using NumPy
def expand_dims(input, axis):
    return np.expand_dims(input, axis)

def cast(x, dtype):
    return np.array(x, dtype=dtype)

def argmax(input, axis=None):
    return np.argmax(input, axis=axis)

class Variable:
    def __init__(self, val):
        self.val = np.array(val)
    
    def numpy(self):
        return self.val

class GradientTape:
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def watch(self, tensor):
        pass
        
    def gradient(self, target, sources):
        # We will compute the analytical gradients directly in our model's Grad-CAM method.
        # This placeholder returns a matrix of ones shaped like the source, which can act as a fallback.
        if hasattr(sources, 'shape'):
            return np.ones(sources.shape, dtype=np.float32)
        return np.ones((1,), dtype=np.float32)
