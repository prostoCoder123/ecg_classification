import os
from os import path as Path
import pickle
import numpy as np
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle

class TrainTestSplitter(object):
    """
    This class provides methods for converting the ecg data (splitted to qrs segments) 
    processed by PtbXlFeaturesExtractor into the final sets for feeding to the ML models.
        
    The final datasets contain data for training, validation and testing of the ML models,
    saved to the files and can be manipulated as:
    x_train, y_train - set of sqr segments and their class labels for training (randomly shuffled)
    x_valid, y_valid - set of sqr segments and their class labels for validation (randomly shuffled)
    x_test, y_test - set of sqr segments and their class labels for testing (randomly shuffled)

    """
    def __init__(self, pkl_file, samples, numclasses) :
        """
        Initialize the class instance with the params:
        :param string pkl_file - the path to the source .pkl file
        :param int samples - the number of samples of each class
        :param int numclasses - the number of classes (patients)
        
        """
        self.min_numclasses = 2
        self.max_numclasses = 828
        self.min_samples = 3
        self.max_samples = 18
        self.min_ratio = 0.6
        self.max_ratio = 0.98
    
        if numclasses < self.min_numclasses or numclasses > self.max_numclasses :
            raise ValueError(f"The value of numclasses must be greater than {min_numclasses} \
            and smaller than {max_numclasses}")

        if samples < self.min_samples or samples > self.max_samples :
            raise ValueError(f"The value of samples must be greater than {min_samples} \
            and smaller than {max_samples}")

        if not Path.exists(pkl_file) :
            raise ValueError(f"The source file {pkl_file} does no exist")
            
        self.path = os.path.dirname(pkl_file) # path to the preprocessed data (X and Y sets)
        self.pkl_file = pkl_file # the name of the file with preprocessed data
        self.samples = samples # the number of samples of each class
        self.numclasses = numclasses # the number of classes that need to get extracted from X, Y
        self.train_file = Path.join(self.path, "train_xy_" + str(numclasses) + ".pkl")
        self.test_file = Path.join(self.path, "test_xy_" + str(numclasses) + ".pkl")
        self.validation_file = Path.join(self.path, "validation_xy_" + str(numclasses) + ".pkl")
        
    def load_preprocessed_data(self) :
        """
        Reads and returns X (qrs segments) and Y (class labels) sets from the source .pkl file
        """
        with open(self.pkl_file, 'rb') as f: x, y = pickle.load(f)
    
        return np.array(x), np.array(y)
    
    
    def get_training_data(self, train_ratio = 0.9) :
        '''
        Split the preprocced X, Y data to the training set, validation set, test set in specified proportions.
        The data is splitted according to the ratio train_ratio:
        - train_ratio * 100% of the source data will be selected to place x_train and y_train
        - (1.0 - train_ratio) / 2.0 * 100% of the source data will be selected to place 
        x_valid, y_valid, x_test, y_test respectively
        
        By default train_ratio = 0.9 that means 90% - 5% - 5% proportions for
        training, validation, test datasets respectively
        
        '''
        self.check_ratio(train_ratio)
        
        if self.check_datasets() == True : # if the processed data exists on the disc - just load and return it
            with open(self.train_file, 'rb') as f : x_train, y_train = pickle.load(f)
            with open(self.test_file, 'rb') as f : x_test, y_test = pickle.load(f)
            with open(self.validation_file, 'rb') as f : x_valid, y_valid = pickle.load(f)
        else : # split the preprocessed data into three target sets
            x, y = self.load_preprocessed_data()
            y = y.astype(int).flatten()

            x_train = []
            x_test = []
            x_valid = []
            y_train = []
            y_test = []
            y_valid = []

            rs = ShuffleSplit(n_splits=2, train_size=train_ratio, test_size=1.0-train_ratio, random_state=82)
            rs_half = ShuffleSplit(n_splits=2, train_size=0.5, test_size=0.5, random_state=54)
            samples_indices = range(self.samples)

            # 90/5/5
            for i, v in enumerate(range(0, self.numclasses * self.samples - 1, self.samples)) :
                patient_samples = x[v:v+self.samples] # take n samples of the class i

                 # get indices of samples for training data and the rest data
                train_index, test_val_index = next(rs.split(samples_indices))
                
                x_train.extend(patient_samples[train_index]) # select the samples and labels for training
                y_train.extend([to_categorical(i, self.numclasses)] * len(train_index)) # one-hot encoding for labels

                x_test_val = patient_samples[test_val_index] # the rest of the samples are splitted to validation and test
                test_index, validation_index = next(rs_half.split(x_test_val))
                
                x_test.extend(x_test_val[test_index])
                y_test.extend([to_categorical(i, self.numclasses)] * len(test_index))
                x_valid.extend(x_test_val[validation_index])
                y_valid.extend([to_categorical(i, self.numclasses)] * len(validation_index))

            with open(self.train_file, 'wb') as f: pickle.dump((x_train, y_train), f)
            with open(self.test_file, 'wb') as f: pickle.dump((x_test, y_test), f)
            with open(self.validation_file, 'wb') as f: pickle.dump((x_valid, y_valid), f)

        x_train, y_train = shuffle(np.asarray(x_train), np.asarray(y_train))
        x_test, y_test = shuffle(np.asarray(x_test), np.asarray(y_test))
        x_valid, y_valid = shuffle(np.asarray(x_valid), np.asarray(y_valid))
        
        return x_train, x_test, x_valid, y_train, y_test, y_valid

    def check_ratio(self, ratio) :
        if ratio < self.min_ratio or ratio > self.max_ratio :
            raise ValueError(f"The ratio must be in [{self.min_ratio}, {self.max_ratio}]")
   
    def check_datasets(self) :
        """
        Check if the datasets are located if the .pkl files
        """
        return Path.exists(self.train_file) and \
           Path.exists(self.test_file) and \
           Path.exists(self.validation_file)
