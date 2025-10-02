from tensorflow.keras import (
    Sequential, 
    Input
)
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomFlip,
    RandomTranslation,
    RandomZoom,
    Rescaling
)
from tensorflow.keras.layers import (
    Dense, 
    Conv2D, 
    MaxPool2D,
    Flatten
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import (
    Precision,
    Recall
)

def get_visem_cnn():
    """
        Builds and compiles a Convolutional Neural Network (CNN) model for image classification.

        The model architecture includes:
            - Input layer for 48x48 RGB images.
            - Data augmentation layers: random flipping, translation, and zoom.
            - Rescaling layer to normalize pixel values.
            - Two convolutional blocks, each with Conv2D and MaxPool2D layers.
            - Flatten layer to convert feature maps to a 1D vector.
            - Three dense layers for feature extraction.
            - Output layers: a dense layer with 1 unit (sigmoid activation) for binary classification.
        The model is compiled with:
            - Adam optimizer (learning rate: 0.00015)
            - Binary crossentropy loss
            - Precision metric
            
        Returns:
            Compiled CNN model.
    """
    model = Sequential([
                Input(shape=(48, 48, 3)),
                RandomFlip(mode='horizontal_and_vertical'),
                RandomTranslation(height_factor=0.1,
                                  width_factor=0.1),
                RandomZoom(height_factor=0.2,
                           width_factor=0.2),
                Rescaling(scale=1./255,
                          offset=0.0),
                
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='same'),
                MaxPool2D(pool_size=(3,3),
                          strides=(2,2)),
                
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='same'),
                MaxPool2D(pool_size=(3,3),
                          strides=(2,2)),
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='same'),
                MaxPool2D(pool_size=(3,3),
                          strides=(2,2)),

                Flatten(),
                
                 Dense(units=32,
                      activation='relu',
                      kernel_initializer='random_normal'),
                Dense(units=32,
                      activation='relu',
                      kernel_initializer='random_normal'),
                Dense(units=4,
                      activation='relu',
                      kernel_initializer='random_normal'),
                
                Dense(units=4,
                      activation='relu',
                      kernel_initializer='random_normal'),
                Dense(units=1,
                      activation='sigmoid',
                      kernel_initializer='random_normal')
            ])

    model.compile(optimizer=Adam(learning_rate=0.00015), 
                  loss=BinaryCrossentropy(), 
                  metrics=[Precision(name="precision", thresholds=0.5),
                           Recall(name="recall", thresholds=0.5)])

    return model