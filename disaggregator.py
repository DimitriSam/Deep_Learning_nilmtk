from __future__ import print_function, division
from warnings import warn, filterwarnings

import random
import sys
import pandas as pd
import numpy as np

from nilmtk.disaggregate import Disaggregator
from nilmtk.datastore import HDFDataStore

from dataset_processing import normalise, inversenormalise

class NeuralDisaggregator(Disaggregator):
    '''Attempt to create a RNN Disaggregator

    Attributes
    ----------
    model : keras Sequential model
    mmax : the maximum value of the aggregate data

    MIN_CHUNK_LENGTH : int
       the minimum length of an acceptable chunk
    '''
    
    def __init__(self, model, window_size=100):
        '''Initialize disaggregator
        '''
        self.MIN_CHUNK_LENGTH = window_size
        self.window_size = window_size
        self.model = model
            
    def disaggregate(self, mains, output_datastore, meter_metadata, **load_kwargs):
        '''Disaggregate mains according to the model learnt previously.

        Parameters
        ----------
        mains : a nilmtk.ElecMeter of aggregate data
        meter_metadata: a nilmtk.ElecMeter of the observed meter used for storing the metadata
        output_datastore : instance of nilmtk.DataStore subclass
            For storing power predictions from disaggregation algorithm.
        **load_kwargs : key word arguments
            Passed to `mains.power_series(**kwargs)`
        '''

        load_kwargs = self._pre_disaggregation_checks(load_kwargs)

        load_kwargs.setdefault('sample_period', 60)
        load_kwargs.setdefault('sections', mains.good_sections())

        timeframes = []
        building_path = '/building{}'.format(mains.building())
        mains_data_location = building_path + '/elec/meter1'
        data_is_available = False

        for chunk in mains.power_series(**load_kwargs):
            if len(chunk) < self.MIN_CHUNK_LENGTH:
                continue
            print("New sensible chunk: {}".format(len(chunk)))

            timeframes.append(chunk.timeframe)
            measurement = chunk.name
            chunk2 = normalise(chunk)

            appliance_power = self.disaggregate_chunk(chunk2)
            appliance_power[appliance_power < 0] = 0
            appliance_power = inversenormalise(appliance_power)

            # Append prediction to output
            data_is_available = True
            cols = pd.MultiIndex.from_tuples([chunk.name])
            meter_instance = meter_metadata.instance()
            df = pd.DataFrame(
                appliance_power.values, index=appliance_power.index,
                columns=cols, dtype="float32")
            key = '{}/elec/meter{}'.format(building_path, meter_instance)
            output_datastore.append(key, df)

            # Append aggregate data to output
            mains_df = pd.DataFrame(chunk, columns=cols, dtype="float32")
            output_datastore.append(key=mains_data_location, value=mains_df)

        # Save metadata to output
        if data_is_available:
            self._save_metadata_for_disaggregation(
                output_datastore=output_datastore,
                sample_period=load_kwargs['sample_period'],
                measurement=measurement,
                timeframes=timeframes,
                building=mains.building(),
                meters=[meter_metadata]
            )

    def disaggregate_chunk(self, mains):
        '''In-memory disaggregation.

        Parameters
        ----------
        mains : pd.Series of aggregate data
        Returns
        -------
        appliance_powers : pd.DataFrame where each column represents a
            disaggregated appliance.  Column names are the integer index
            into `self.model` for the appliance in question.
        '''
        up_limit = len(mains)

        mains.fillna(0, inplace=True)

        X_batch = np.array(mains)
        Y_len = len(X_batch)
        indexer = np.arange(self.window_size)[None, :] + np.arange(len(X_batch)-self.window_size+1)[:, None]
        X_batch = X_batch[indexer]
        X_batch = np.reshape(X_batch, (X_batch.shape[0],X_batch.shape[1],1))

        pred = self.model.predict(X_batch, batch_size=128)
        pred = np.reshape(pred, (len(pred)))
        column = pd.Series(pred, index=mains.index[self.window_size-1:Y_len], name=0)

        appliance_powers_dict = {}
        appliance_powers_dict[0] = column
        appliance_powers = pd.DataFrame(appliance_powers_dict)
        return appliance_powers
