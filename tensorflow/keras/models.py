import json
import numpy as np
from . import layers

class Sequential:
    def __init__(self, layers_list=None):
        self.layers = layers_list or []
        
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, optimizer='adam', loss='categorical_crossentropy', metrics=None):
        pass
        
    def __call__(self, x):
        return self.predict(x)
        
    def predict(self, x):
        # Ensure batch dimension
        if len(x.shape) == 3:
            x = np.expand_dims(x, axis=0)
        
        out = x
        for layer in self.layers:
            out = layer(out)
        return out
        
    def save(self, filepath):
        import os
        data = {}
        config = []
        for i, layer in enumerate(self.layers):
            layer_config = {
                'class_name': layer.__class__.__name__,
                'config': {
                    'name': layer.name,
                }
            }
            if hasattr(layer, 'filters'):
                layer_config['config']['filters'] = layer.filters
                layer_config['config']['kernel_size'] = layer.kernel_size
                layer_config['config']['activation'] = layer.activation
                layer_config['config']['padding'] = getattr(layer, 'padding', 'same')
            if hasattr(layer, 'pool_size'):
                layer_config['config']['pool_size'] = layer.pool_size
                layer_config['config']['strides'] = layer.strides
            if hasattr(layer, 'units'):
                layer_config['config']['units'] = layer.units
                layer_config['config']['activation'] = layer.activation
                
            config.append(layer_config)
            
            # Save weights
            for j, w in enumerate(layer.weights):
                data[f'layer_{i}_weight_{j}'] = w
                
        data['config_json'] = json.dumps(config)
        
        # NumPy's np.savez always appends .npz to the name if not present
        if filepath.endswith('.keras'):
            np.savez(filepath, **data)
            actual_saved = filepath + '.npz'
            if os.path.exists(actual_saved):
                if os.path.exists(filepath):
                    os.remove(filepath)
                os.rename(actual_saved, filepath)
        else:
            np.savez(filepath, **data)

def load_model(filepath):
    import os
    if not os.path.exists(filepath) and os.path.exists(filepath + '.npz'):
        filepath = filepath + '.npz'
        
    archive = np.load(filepath)
    config_json = str(archive['config_json'])
    config = json.loads(config_json)
    
    model = Sequential()
    for i, layer_config in enumerate(config):
        class_name = layer_config['class_name']
        kwargs = layer_config['config']
        
        if class_name == 'Conv2D':
            layer = layers.Conv2D(
                filters=kwargs['filters'],
                kernel_size=kwargs['kernel_size'],
                activation=kwargs['activation'],
                name=kwargs['name'],
                padding=kwargs.get('padding', 'same')
            )
        elif class_name == 'MaxPooling2D':
            layer = layers.MaxPooling2D(
                pool_size=kwargs['pool_size'],
                strides=kwargs['strides'],
                name=kwargs['name']
            )
        elif class_name == 'Flatten':
            layer = layers.Flatten(name=kwargs['name'])
        elif class_name == 'GlobalAveragePooling2D':
            layer = layers.GlobalAveragePooling2D(name=kwargs['name'])
        elif class_name == 'Dense':
            layer = layers.Dense(
                units=kwargs['units'],
                activation=kwargs['activation'],
                name=kwargs['name']
            )
        else:
            raise ValueError(f"Unknown layer type: {class_name}")
            
        weights = []
        j = 0
        while f'layer_{i}_weight_{j}' in archive:
            weights.append(archive[f'layer_{i}_weight_{j}'])
            j += 1
            
        layer.set_weights(weights)
        model.add(layer)
        
    return model
