#Adapted from https://github.com/amorten/openlis/blob/master/openlis/data/data_set.py

import numpy as np


class DataSet(object):
    def __init__(self, keys, positions=None, num_positions=None):
        """Construct a DataSet.
        positions = labels of where keys should go
        """
        assert keys.shape[0] == positions.shape[0], (
                    'keys.shape: %s positions.shape: %s' % (keys.shape, positions.shape))

        self._num_keys = keys.shape[0]
        
        
        self._keys = np.array(keys)
        if positions is not None:
                self._positions = np.array(positions)
        else:
                self._keys = np.sort(keys)
                self._positions = np.arange(self._num_keys)

        if num_positions is not None:
            self._num_positions = num_positions
        else:
            if len(self._positions) == 0:
                self._num_positions = 0
            else:
                self._num_positions = self._positions[-1] + 1
            
        self._epochs_completed = 0
        self._index_in_epoch = 0

        if len(keys.shape) > 1:
            self._key_size = keys.shape[1]
        else:
            self._key_size = 1

        if len(keys) > 0:
            self._keys_mean = np.mean(keys)
            self._keys_std = np.std(keys)
        else:
            self._keys_mean = None
            self._keys_std = None
            
    @property
    def keys(self):
        return self._keys

    @property
    def positions(self):
        return self._positions

    @property
    def num_keys(self):
        return self._num_keys

    @property
    def num_positions(self):
        return self._num_positions

    @property
    def epochs_completed(self):
        return self._epochs_completed

    @property
    def key_size(self):
        return self._key_size

    @property
    def keys_mean(self):
        return self._keys_mean

    @property
    def keys_std(self):
        return self._keys_std
    
    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_keys:
            # Finished epoch
            self._epochs_completed += 1
            if shuffle:
                # Shuffle the data
                perm = np.arange(self._num_keys)
                np.random.shuffle(perm)
                self._keys = self._keys[perm]
                self._positions = self._positions[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_keys
        end = self._index_in_epoch
        return self._keys[start:end], self._positions[start:end]

    def reset_epoch(self):
        self._index_in_epoch = 0
        

def create_train_validate_test_data_sets(data_set, val_ratio, test_ratio):
    """Creates training and validation data sets.
    """
    #Shuffle the keys and positions by same permutation
    perm = np.arange(data_set.num_keys)
    np.random.shuffle(perm)
    keys = data_set.keys[perm]
    positions = data_set.positions[perm]
    num_val = int(data_set.num_keys * val_ratio)
    num_test = int(data_set.num_keys * test_ratio)
    num_train = data_set.num_keys - num_val - num_test

    train_keys = keys[:num_train]
    train_positions = positions[:num_train]
    validation_keys = keys[num_train : (num_train+num_val)]
    validation_positions = positions[num_train : (num_train+num_val)]
    test_keys = keys[(num_train+num_val):]
    test_positions = positions[(num_train+num_val):]

    train = DataSet(np.reshape(train_keys,[-1,1]), train_positions)
    validation = DataSet(validation_keys, validation_positions)
    test = DataSet(test_keys, test_positions)

    class DataSets(object):
        pass

    data_sets = DataSets()
    data_sets.train = train
    data_sets.validate = validation
    data_sets.test = test
    return data_sets


def load_synthetic_data(data_path):
    #assume data from txt file is already sorted
    keys = []
    with open(data_path, 'r') as file:
        for line in file:
            keys.append(float(line.strip()))
    num_keys = len(keys)
    keys = np.array(keys)
    positions = np.arange(num_keys).astype(float)
    positions /= num_keys #normalize so that values lie in range [0, 1]
    return DataSet(keys=keys, positions=positions)
