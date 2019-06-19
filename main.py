from __future__ import print_function, division
import time


from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
#from windowgrudisaggregator import WindowGRUDisaggregator
from dataset_processing import load_dataset, data_processing
from batch_generator import Batch_Generator
from models import GRU_model, RNN_model, DAE_model, DresNET_model
from disaggregator import NeuralDisaggregator
from keras.models import load_model
import metrics

from keras.callbacks import ModelCheckpoint




# =====Define paramaters======

info = {'filename': 'ukdale.h5',
        'meter_label': 'kettle',  # ["kettle" , "microwave" , "dishwasher" , "fridge" , "washing_machine"]
        'train_building': [1],
        'test_building': 1,
        'sample_period': 6,
        'start_train': '13-4-2013', 'end_train': '13-6-2013',  # Define the time intervals of training and test data
        'start_test': '1-1-2014', 'end_test': '30-6-2014'
        }

# Parameters
params = {'batch_size': 128,
          'window_size': 100,
          'model_name': 'GRU',
          'shuffle': False}


# =====Load Dataset======
train_meterlist, train_mainlist, test_meterlist, test_mainlist = load_dataset(**info)

train_x, train_y = data_processing(train_mainlist, train_meterlist, window_size=100, sample_period=6)

# #Batch generator
# gen = batch_generator(train_x, train_y,batch_size=128)

# Batch generator
gen = Batch_Generator(**params)
t = gen.generator(train_x, train_y)

if params['model_name'] == 'LSTM':
    model = RNN_model(params['window_size'])

elif params['model_name'] == 'GRU':
    model = GRU_model(params['window_size'])

elif params['model_name'] == 'DAE':
    model = DAE_model(params['window_size'])

elif params['model_name'] == 'DresNET':
    model = DresNET_model(params['window_size'])


# Training
filepath_checkpoint = "UKDALE-RNN-h " + str(info['train_building']) + str(info['meter_label']) + ' epo.hdf5'
filepath = 'UKDALE-RNN-h1-kettle-5epochs.h5'


mode = 'load_pretrained'

if mode == 'training':

    print("*********Training*********")
    start = time.time()

    checkpointer = ModelCheckpoint(filepath_checkpoint,
                                   verbose=1, save_best_only=True)
    model.fit_generator(t, steps_per_epoch=12839, epochs=1, callbacks=[checkpointer])

    end = time.time()
    print('### Total trainning time cost: {} ###'.format(str(end - start)))

elif mode == 'loading':
    # load checkpoints weights
    print(filepath_checkpoint)
    model.load_weights(filepath_checkpoint)
    
elif mode == 'load_pretrained':
    #load pretrained model
    model = load_model(filepath)

print("*********Disaggregate*********")
disaggregator = NeuralDisaggregator(model)
disag_filename = "disag-out.h5"
output = HDFDataStore(disag_filename, 'w')
disaggregator.disaggregate(test_mainlist, output, test_meterlist, sample_period = info['sample_period'])
output.close()