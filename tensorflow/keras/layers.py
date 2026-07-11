import numpy as np

class Layer:
    def __init__(self):
        self.weights = []
        self.name = None
        
    def __call__(self, x):
        return self.forward(x)
        
    def forward(self, x):
        return x
        
    def set_weights(self, weights):
        self.weights = [np.array(w) for w in weights]
        
    def get_weights(self):
        return self.weights

class Conv2D(Layer):
    def __init__(self, filters, kernel_size, activation=None, input_shape=None, name=None, padding='same'):
        super().__init__()
        self.filters = filters
        if isinstance(kernel_size, (list, tuple)):
            self.kernel_size = tuple(kernel_size)
        else:
            self.kernel_size = (kernel_size, kernel_size)
        self.activation = activation
        self.name = name
        self.padding = padding.lower()
        self.weights = [] # [kernel, bias]
        
    def forward(self, x):
        # Input shape: (N, H, W, C)
        # weights[0] (kernel): (kh, kw, in_c, out_c)
        # weights[1] (bias): (out_c,)
        kernel = self.weights[0]
        bias = self.weights[1]
        
        N, H, W, C = x.shape
        kh, kw, in_c, out_c = kernel.shape
        
        # If 1x1 convolution
        if kh == 1 and kw == 1:
            x_flat = x.reshape(-1, C)
            out_flat = np.dot(x_flat, kernel.squeeze((0, 1))) + bias
            out = out_flat.reshape(N, H, W, out_c)
        else:
            # 3x3 convolution with 'same' padding
            if self.padding == 'same':
                pad_h = kh // 2
                pad_w = kw // 2
                x_padded = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)), mode='constant')
            else:
                x_padded = x
                
            padded_h, padded_w = x_padded.shape[1], x_padded.shape[2]
            out_h = padded_h - kh + 1
            out_w = padded_w - kw + 1
            
            out = np.zeros((N, out_h, out_w, out_c), dtype=np.float32)
            for i in range(out_h):
                for j in range(out_w):
                    patch = x_padded[:, i:i+kh, j:j+kw, :]
                    out[:, i, j, :] = np.einsum('nhwc,hwco->no', patch, kernel) + bias
                    
        if self.activation == 'relu':
            out = np.maximum(out, 0)
        return out

class MaxPooling2D(Layer):
    def __init__(self, pool_size=(2, 2), strides=None, name=None):
        super().__init__()
        if isinstance(pool_size, (list, tuple)):
            self.pool_size = tuple(pool_size)
        else:
            self.pool_size = (pool_size, pool_size)
            
        strides = strides or self.pool_size
        if isinstance(strides, (list, tuple)):
            self.strides = tuple(strides)
        else:
            self.strides = (strides, strides)
        self.name = name
        
    def forward(self, x):
        N, H, W, C = x.shape
        ph, pw = self.pool_size
        sh, sw = self.strides
        
        out_h = (H - ph) // sh + 1
        out_w = (W - pw) // sw + 1
        
        out = np.zeros((N, out_h, out_w, C), dtype=np.float32)
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * sh
                h_end = h_start + ph
                w_start = j * sw
                w_end = w_start + pw
                out[:, i, j, :] = np.max(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        return out

class Flatten(Layer):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        
    def forward(self, x):
        N = x.shape[0]
        return x.reshape(N, -1)

class GlobalAveragePooling2D(Layer):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        
    def forward(self, x):
        return np.mean(x, axis=(1, 2))

class Dense(Layer):
    def __init__(self, units, activation=None, name=None):
        super().__init__()
        self.units = units
        self.activation = activation
        self.name = name
        self.weights = [] # [kernel, bias]
        
    def forward(self, x):
        kernel = self.weights[0]
        bias = self.weights[1]
        
        out = np.dot(x, kernel) + bias
        if self.activation == 'relu':
            out = np.maximum(out, 0)
        elif self.activation == 'softmax':
            shift_x = out - np.max(out, axis=-1, keepdims=True)
            exps = np.exp(shift_x)
            out = exps / np.sum(exps, axis=-1, keepdims=True)
        return out
