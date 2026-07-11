from .models import Sequential
from . import layers

def MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet'):
    # Create a simple mock Conv2D base model that matches the applications signature
    model = Sequential()
    # Conv1: 3x3 filter to extract color/texture features
    model.add(layers.Conv2D(8, 3, activation='relu', input_shape=input_shape, name='conv_1'))
    # MaxPool down to 56x56
    model.add(layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), name='pool_1'))
    # Conv2: last conv layer for Grad-CAM
    model.add(layers.Conv2D(16, 3, activation='relu', name='conv_2'))
    # Global average pooling
    model.add(layers.GlobalAveragePooling2D(name='global_average_pooling2d'))
    return model
