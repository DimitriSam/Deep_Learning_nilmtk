import numpy as np


class Batch_Generator():

    def __init__(self, batch_size, window_size, model_name, shuffle=True):

        self.name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.window_size = window_size
        
    def num_epochs(self, inputs):
        
        batch_size = int(self.batch_size / len(inputs))
        num_of_batches = [(int((len(inputs[i])-self.window_size+1) / batch_size) - 1) for i in range(len(inputs))]
        n_epochs = min(num_of_batches)
        
        return n_epochs

    def generator(self, inputs, targets):
        
        while True:
            
            num_meters = len(inputs)
            batch_size = int(self.batch_size / num_meters)
            #num_of_batches = [(int(len(inputs[i]) / batch_size) - 1) for i in range(len(inputs))]

            n_epochs = self.num_epochs(inputs)


            # Batch indexes
            self.indexes = list(range(n_epochs))

            if self.shuffle == True:
                np.random.shuffle(self.indexes)

            for ei, e in enumerate(self.indexes):

                offset = e * batch_size

                # Initialization
                X_batch = np.zeros((batch_size * num_meters, self.window_size, 1))  # (128,100,1)
                Y_batch = np.zeros((batch_size * num_meters, 1))  # (128,1)

                # Create a batch out of data from all buildings
                for i in range(num_meters):
                    mainpart = inputs[i]
                    meterpart = targets[i]

                    indexer = np.arange(self.window_size)[None, :] + np.arange(len(inputs[i])-self.window_size+1)[offset:offset + batch_size, None]

                    mainpart = mainpart[indexer]
                    meterpart = meterpart[self.window_size - 1:][offset:offset + batch_size]

                    X = np.reshape(mainpart, (batch_size, self.window_size, 1))
                    Y = np.reshape(meterpart, (batch_size, 1))

                    X_batch[i * batch_size:(i + 1) * batch_size] = np.array(X)
                    Y_batch[i * batch_size:(i + 1) * batch_size] = np.array(Y)

                # Shuffle data
                p = np.random.permutation(len(X_batch))
                X_batch, Y_batch = X_batch[p], Y_batch[p]


                yield X_batch, Y_batch