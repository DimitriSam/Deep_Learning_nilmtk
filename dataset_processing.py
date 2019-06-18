from __future__ import print_function, division
import time

from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore


import random
import sys
import pandas as pd
import numpy as np
import h5py



def load_dataset(filename, meter_label, train_building, test_building,
                 sample_period, start_train, end_train, start_test, end_test):
    train = DataSet(filename)
    test = DataSet(filename)

    train.set_window(start=start_train, end=end_train)
    test.set_window(start=start_test, end=end_test)

    # if only onw house is used for training
    # train_y = train.buildings[train_building].elec[meter_label]
    # train_x = train.buildings[train_building].elec.mains()

    # multiple houses for training
    train_meterlist = [train.buildings[i].elec[meter_label] for i in train_building]
    train_mainlist = [train.buildings[i].elec.mains() for i in train_building]

    test_meterlist = test.buildings[test_building].elec[meter_label]
    test_mainlist = test.buildings[test_building].elec.mains()

    assert len(train_mainlist) == len(train_meterlist), "The number of main and apliances meters must be equal"

    return train_meterlist, train_mainlist, test_meterlist, test_mainlist


def data_processing(train_mainlist, train_meterlist, window_size, **load_kwargs):
    '''Data processing

    Parameters
    ----------
    train_mainlist : a list of nilmtk.ElecMeter objects for the aggregate data of each building
    train_meterlist : a list of nilmtk.ElecMeter objects for the meter data of each building
    '''

    train_x = [next(m.power_series(**load_kwargs)) for m in train_mainlist]
    train_y = [next(m.power_series(**load_kwargs)) for m in train_meterlist]

    # mmax = max([m.max() for m in train_x])
    # Normalize the data
    train_x = [normalise(data) for data in train_x]
    train_y = [normalise(data) for data in train_y]

    # replca NaN values and
    for i in range(len(train_x)):
        train_x[i].fillna(0, inplace=True)
        train_y[i].fillna(0, inplace=True)
        ix = train_x[i].index.intersection(train_y[i].index)
        # m1 = train_x[i]
        # m2 = train_y[i]
        # train_x[i] = m1[ix]
        # train_y[i] = m2[ix]
        train_x[i] = train_x[i][ix]
        train_y[i] = train_y[i][ix]

        indexer = np.arange(window_size)[None, :] + np.arange(len(train_x[i].values) - window_size + 1)[:, None]
        train_x[i] = train_x[i].values[indexer]
        train_y[i] = train_y[i].values[window_size - 1:]

    return train_x, train_y


# def batch_generator(mainchunks,meterchunks,batch_size=128):
#     num_of_batches = [(int(len(mainchunks[i])/batch_size) - 1) for i in range(len(mainchunks))]
#     num_meters = len(mainchunks)
#     batch_size = int(batch_size/num_meters)
#     window_size = 100

#     batch_indexes = list(range(min(num_of_batches))) #[1,2,3.....,12839]
#     random.shuffle(batch_indexes)  #[123,5,456,4,2.....]

#     for bi, b in enumerate(batch_indexes): # Iterate for every batch
#         X_batch = np.zeros((batch_size*num_meters, window_size, 1)) #(128,100,1)
#         Y_batch = np.zeros((batch_size*num_meters, 1))              #(128,1)


#         # Create a batch out of data from all buildings
#         for i in range(num_meters):
#             mainpart = mainchunks[i]
#             meterpart = meterchunks[i]
#             mainpart = mainpart[b*batch_size:(b+1)*batch_size]
#             meterpart = meterpart[b*batch_size:(b+1)*batch_size]
#             X = np.reshape(mainpart, (batch_size, window_size, 1))
#             Y = np.reshape(meterpart, (batch_size, 1))

#             X_batch[i*batch_size:(i+1)*batch_size] = np.array(X)
#             Y_batch[i*batch_size:(i+1)*batch_size] = np.array(Y)

#         # Shuffle data
#         p = np.random.permutation(len(X_batch))
#         X_batch, Y_batch = X_batch[p], Y_batch[p]
#         yield X_batch,Y_batch


def normalise(data):
    """
    Perform the normalisation (x-min(x))/(max(x)-min(x)).
    --------------------------------------
    :arg
    data: data that needs to be transformed
    mean: mean value of data
    max_v: max value of data
    :return
    The normalized data
    """
    mean = data.mean()
    max_v = data.max()

    # return data/max_v #remember to try this normalization as well
    return (data - mean) / (max_v - mean)


def inversenormalise(data):
    """
    Perform the in-normalisation data*(max(x)-min(x))+min(x).
    ------------------------------------------------
    :arg
    data: data that needs to be inverse-transformed
    mean: mean value of data
    max_v: max value of data
    :return
    The in-normalized data
    """

    mean = data.mean()
    max_v = data.max()

    return data * (max_v - mean) + mean


def _create_model(self):
    '''Creates the GRU architecture described in the paper
    '''
    model = Sequential()

    # 1D Conv
    model.add(Conv1D(16, 4, activation='relu', input_shape=(self.window_size, 1), padding="same", strides=1))

    # Bi-directional GRUs
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
