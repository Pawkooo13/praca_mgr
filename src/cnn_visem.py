from tensorflow.keras import (
    Sequential, 
    Input
)
from tensorflow.keras.layers.experimental.preprocessing import (
    RandomFlip,
    RandomTranslation,
    RandomZoom,
    Rescaling,
    RandomRotation
)
from tensorflow.keras.layers import (
    Dense, 
    Conv2D, 
    MaxPool2D,
    Flatten,
    BatchNormalization,
    Dropout
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
            - Input layer for 32x32 RGB images.
            - Data augmentation layers: random flipping, translation, and zoom.
            - Rescaling layer to normalize pixel values.
            - Two convolutional blocks, each with Conv2D and MaxPool2D layers.
            - Flatten layer to convert feature maps to a 1D vector.
            - Three dense layers for feature extraction.
            - Output layers: a dense layer with 1 unit (sigmoid activation) for binary classification.
        The model is compiled with:
            - Adam optimizer (learning rate: 0.0001)
            - Binary crossentropy loss
            - Precision metric
            
        Returns:
            Compiled CNN model.
    """
    model = Sequential([
                Input(shape=(32, 32, 3)),
                RandomFlip(mode='horizontal_and_vertical'),
                RandomTranslation(height_factor=0.2,
                                  width_factor=0.2),
                RandomZoom(height_factor=0.2,
                           width_factor=0.2),
                RandomRotation(factor=0.2),
                Rescaling(scale=1./255,
                          offset=0.0),
                
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='valid'),
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='valid'),
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2),
                          strides=(2,2)),
                
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='valid'),
                Conv2D(filters=8,
                       kernel_size=(3,3),
                       kernel_initializer='random_normal',
                       strides=(1,1),
                       activation='relu',
                       padding='valid'),       
                BatchNormalization(),
                MaxPool2D(pool_size=(2,2),
                          strides=(2,2)),
            
                Flatten(),
                
                Dense(units=16,
                      activation='relu',
                      kernel_initializer='random_normal'),
                Dropout(0.3),
                Dense(units=1,
                      activation='sigmoid',
                      kernel_initializer='random_normal')
            ])

    print(model.summary())

    model.compile(optimizer=Adam(learning_rate=0.000015), 
                  loss=BinaryCrossentropy(), 
                  metrics=[Precision(name="precision", thresholds=0.5),
                           Recall(name="recall", thresholds=0.5)])

    return model