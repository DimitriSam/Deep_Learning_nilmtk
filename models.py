from keras.models import load_model, Model, Input
from keras.layers import Dense, Conv1D, GRU, LSTM, Bidirectional, Dropout,Conv2D
from keras.layers import Reshape, BatchNormalization, Activation, Flatten, Concatenate
from keras.models import Sequential

def RNN_model(window_size):
    '''Creates the RNN module described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation="linear", input_shape=(window_size,1), padding="same", strides=1))

    #Bi-directional LSTMs
    model.add(Bidirectional(LSTM(128, return_sequences=True, stateful=False), merge_mode='concat'))
    model.add(Bidirectional(LSTM(256, return_sequences=False, stateful=False), merge_mode='concat'))

    # Fully Connected Layers
    model.add(Dense(128, activation='tanh'))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    #plot_model(model, to_file='model.png', show_shapes=True)

    return model


def GRU_model(window_size):

    '''Creates the GRU architecture described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation='relu', input_shape=(window_size,1), padding="same", strides=1))

    #Bi-directional GRUs
    model.add(Bidirectional(GRU(64, activation='relu', return_sequences=True), merge_mode='concat'))
    model.add(Dropout(0.5))
    model.add(Bidirectional(GRU(128, activation='relu', return_sequences=False), merge_mode='concat'))
    model.add(Dropout(0.5))

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    print(model.summary())

    return model


def DAE_model(window_size):
    '''Creates and returns the ShortSeq2Point Network
     Based on: https://arxiv.org/pdf/1612.09106v3.pdf
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(30, 10, activation='relu', input_shape=(window_size,1), padding="same", strides=1))
    model.add(Dropout(0.5))
    model.add(Conv1D(30, 8, activation='relu', padding="same", strides=1))
    model.add(Dropout(0.5))
    model.add(Conv1D(40, 6, activation='relu', padding="same", strides=1))
    model.add(Dropout(0.5))
    model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
    model.add(Dropout(0.5))
    model.add(Conv1D(50, 5, activation='relu', padding="same", strides=1))
    model.add(Dropout(0.5))
    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    model.compile(loss='mse', optimizer='adam')
    print(model.summary())

    return model

def DresNET_model(window_size):
    '''Creates the GRU architecture described in the paper
    '''
    def residual_block(filters,x,stride = 1,dilate = None):
        resiual = x
        out = BatchNormalization()(x)
        out1 = Activation('relu')(out)
        out = Conv1D(filters = filters,kernel_size = [3],dilation_rate = dilate,strides = [1],padding = 'same')(out1)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv1D(filters = filters,kernel_size = [3],strides = [1],padding = 'same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)
        out = Conv1D(filters = filters,kernel_size = [1],dilation_rate = dilate,strides = [1],padding = 'same')(out)

        if out1.shape[-1] != filters or stride == 1:
            residual = Conv1D(filters = filters,kernel_size = [3],strides = [1],padding = 'same')(out1)
            out = Concatenate()([residual,out])
        return out

    x = Input(shape = [window_size,1])
    conv1 = Conv1D(filters = 30,kernel_size = [5],dilation_rate = [1],strides = [1],padding = 'same')(x)
    bn = BatchNormalization()(conv1)
    out = Activation('relu')(bn)
    repetition = [3,4,6,3]
    filter_num = [30,40,50,50]
    dilations = [[1],[2],[3],[3]]
    for i in range(len(repetition)):
        for j in range(repetition[i]):
            out = residual_block(filters = filter_num[i],dilate = dilations[i],x = out)

    out = Flatten()(out)
    out = Dense(units = 1)(out)
    model = Model(x,out)
    model.compile(optimizer = 'adam',loss = 'mse')

    print(model.summary())

    return model
