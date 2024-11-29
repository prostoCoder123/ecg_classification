import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import normalize

def build_resnet_model(input_shape, numclasses):
    """
    """
    n_feature_maps = 120
    
    x = Input(shape=(input_shape))
    conv_x = BatchNormalization()(x)
    conv_x = Conv2D(n_feature_maps, 8, 1, padding='same')(conv_x)
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)

    conv_y = Conv2D(n_feature_maps, 5, 1, padding='same')(conv_x)
    conv_y = BatchNormalization()(conv_y)
    conv_y = Activation('relu')(conv_y)

    conv_z = Conv2D(n_feature_maps, 3, 1, padding='same')(conv_y)
    conv_z = BatchNormalization()(conv_z)
     
    is_expand_channels = not (input_shape[-1] == n_feature_maps)
    if is_expand_channels:
        shortcut_y = Conv2D(n_feature_maps, 1, 1,padding='same')(x)
        shortcut_y = BatchNormalization()(shortcut_y)
    else:
        shortcut_y = BatchNormalization()(x)

    y = Add()([shortcut_y, conv_z]) # merging skip connection
    y = Activation('relu')(y)

    full = GlobalAveragePooling2D()(y)
    full = Flatten()(full)
    out = Dense(numclasses, activation='softmax')(full)

    return x, out


def fit_resnet_model(x_train, y_train, x_valid, y_valid, numclasses, input_shape, saved_model_path) :
    '''
    load data, compile and train ResNet1d model, apply data shape trasformation for ANN inputs
    Parameters
    Input: 
        x_train, y_train - train data: qrs segments and labels
        x_valid, y_valid - validation data: qrs segments and labels
        numclasses - the number of classes (labels)
        input_shape - the input shape of the train/validation data
        saved_model_path - path to save the model
    Output: 
        model - resnet model
        history - training history parameters
    '''
    epochs = 100
    batch_size = 4

    x_train, x_valid = map(lambda x: get_resnet_input(x), [x_train, x_valid])

    x, y = build_resnet_model(x_train.shape[1:], numclasses)
    model = Model(inputs=x, outputs=y)

    # create the weights
    model.compile(loss='categorical_crossentropy', # multiclass, singe-label classification
                  optimizer=Adam(learning_rate=0.008),
                  metrics=['accuracy'])
    
    callbacks = [ 
            ModelCheckpoint(filepath=saved_model_path,
                            monitor='val_loss',
                            save_best_only=True),
            #EarlyStopping(monitor="val_accuracy", # Monitor validation accuracy
                          #patience=5) # Interrupts training when accuracy has stopped improving for 5 epochs
        ]

    # show the model's layers summary
    model.summary()
    keras.utils.plot_model(model, "resnet" + ".png", show_shapes=True)

    # train the model
    history = model.fit(x_train, y_train,
                        validation_data=(x_valid, y_valid),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1)

    return model, history

def get_resnet_input(x):
    x = normalize(x)
    return np.reshape(x, (x.shape[0], x.shape[1], 1, 1))

def get_config(self):
    config = super().get_config()
    return config