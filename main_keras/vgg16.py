import os
import sys

cwd = os.getcwd()
print 'current dir: ', cwd
root_path = os.path.abspath(os.path.join(cwd, os.pardir))  # get parent path
print 'root path: ', root_path  
sys.path.insert(0, root_path)


from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Reshape, Input, GRU
from keras.layers import Input, LSTM, Embedding, Dense, TimeDistributed
from keras.models import Model, Sequential


def VGG_LSTM2(weights_path=None): 
    model = Sequential()
    # block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    # final block
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    
    # reshape 4096 = 64 * 64
    model.add(Reshape((64, 64)))
     
    # stack lstm 
    model.add(LSTM(256, return_sequences=True, name='lstm1'))
    model.add(LSTM(256, return_sequences=False, name='lstm2'))

    model.add(Dense(512, activation='relu'))
    model.add(Dense(7)) # regress pose

    if weights_path:
        model.load_weights(weights_path)

    return model



