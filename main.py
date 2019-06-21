from __future__ import print_function, division
import time


from matplotlib import rcParams
import matplotlib.pyplot as plt

from nilmtk import DataSet, TimeFrame, MeterGroup, HDFDataStore
from disaggregator import NeuralDisaggregator
from dataset_processing import load_dataset, data_processing
from batch_generator import Batch_Generator
from models import GRU_model, RNN_model, DAE_model, DresNET_model
from keras.models import load_model
import metrics

from keras.callbacks import ModelCheckpoint




# =====Define paramaters======

info = {'filename': 'drive/My Drive/Dissertation/ukdale.h5',
        'meter_label': 'kettle',  # ["kettle" , "microwave" , "dishwasher" , "fridge" , "washing_machine"]
        'train_building': [1,2],
        'test_building': 5,
        'sample_period': 6
       }

# Parameters
params = {'batch_size': 128,
          'window_size': 100,
          'model_name': 'GRU',
          'shuffle': False}


# =====Load Dataset======
train_meterlist, train_mainlist, test_meterlist, test_mainlist = load_dataset(**info)

train_x, train_y = data_processing(train_mainlist, train_meterlist, window_size=params['window_size'])

# #Batch generator
# gen = batch_generator(train_x, train_y,batch_size=128)


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
filepath = 'drive/My Drive/Dissertation/UKDALE-RNN-h1-kettle-5epochs.h5'


# Batch generator
gen = Batch_Generator(**params)
t = gen.generator(train_x, train_y)
steps_epochs = gen.num_epochs(train_x)


mode = 'training'

if mode == 'training':

    print("*********Training*********")
    start = time.time()

    checkpointer = ModelCheckpoint(filepath_checkpoint,
                                   verbose=1, save_best_only=True)
    model.fit_generator(t, 
                        steps_per_epoch = steps_epochs, 
                        epochs= 1,
                        use_multiprocessing=True,
                        workers=6, 
                        callbacks=[checkpointer])

    
    
    model.save("UKDALE-{}-{}-{}epochs.h5".format(params['model_name'], 
                                                     info['meter_label'],
                                                      1))
    
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
disaggregator = NeuralDisaggregator(model, name = params['model_name'])
disag_filename = "disag-out.h5"
output = HDFDataStore(disag_filename, 'w')
disaggregator.disaggregate(test_mainlist, output, test_meterlist, sample_period = info['sample_period'])
output.close()


print("========== RESULTS ============")
meter_key = info['meter_label']
result = DataSet(disag_filename)
res_elec = result.buildings[info['test_building'].elec
rpaf = metrics.recall_precision_accuracy_f1(res_elec[meter_key], test_mainlist)
print("============ Recall: {}".format(rpaf[0]))
print("============ Precision: {}".format(rpaf[1]))
print("============ Accuracy: {}".format(rpaf[2]))
print("============ F1 Score: {}".format(rpaf[2]))

print("============ Relative error in total energy: {}".format(metrics.relative_error_total_energy(res_elec[meter_key], test_mainlist)))
print("============ Mean absolute error(in Watts): {}".format(metrics.mean_absolute_error(res_elec[meter_key], test_mainlist)))