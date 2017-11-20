import numpy as np
import pandas as pd
class TrainingDataSet(object):

  def __init__(self, x, y):
    #Construct a DataSet for training data.
    assert x.shape[0] == y.shape[0], (
          'size of x: %s size of y: %s' % (x.shape[0], y.shape[0]))

    self._n = x.shape[0]
    self._x = x
    self._y = y
    self._epochs_completed = 0
    self._loc_in_epoch = 0

  @property
  def x(self):
    return self._x

  @property
  def y(self):
    return self._y

  @property
  def n(self):
    return self._n

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    #Return the next `batch_size` worth of data from this data set.
    start = self._loc_in_epoch
    self._loc_in_epoch += batch_size
    if self._loc_in_epoch > self._n:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      index = np.arange(self._n)
      np.random.shuffle(index)
      self._x = self._x[index]
      self._y = self._y[index]
      # Start next epoch
      start = 0
      self._loc_in_epoch = batch_size
      assert batch_size <= self._n
    end = self._loc_in_epoch
    return self._x[start:end], self._y[start:end]

#didn't use
def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    f = gzip.open('./data/organize_data', 'rb')
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

#didn't use
def dense_to_one_hot(labels_dense, num_classes):
  #Convert class labels from scalars to one-hot vectors.
  #.shape =(n rows, m columns)
  #.shape[0] = n or # of rows
  num_labels = labels_dense.shape[0]
  #.arrange = spaced values within given integer
  # = array([0,1,..num_labels]) * num_classes to every value in array 
  index_offset = np.arange(num_labels) * num_classes
  #.zeros(rows,columns) = return 0 array with num_label rows & num_classes column 
  labels_one_hot = np.zeros((num_labels, num_classes))
  #makes array into 1D and makes all values = 1... why?
  labels_one_hot.flat[index_offset + labels_dense.astype(int).ravel()] = 1
  #return array of 1
  return labels_one_hot

def read_data(whole_data_file):
  #filter & grab the columns I want then batch normalize that data set!
  #filename_queue = tf.train.string_input_producer(["file0.csv", "file1.csv"])
  #reader = tf.TextLineReader()
  #key, value = reader.read(filename_queue)

  training_y = pd.read_csv(whole_data_file, usecols=["date","next_return_1","next_return_2","next_return_3","next_return_4","next_return_5","next_return_6","next_return_7","next_return_8","next_return_9","next_return_10"])
  #reshape values from -1 to 1... problem... how does it know what to value?
  #training_y = [training_data['next_return_1'].values
  # training_y = dense_to_one_hot(training_y, 2)
  #drop 1 column 'target' ???   <--- will need to change
  training_x = pd.read_csv("./data/return-data.csv", usecols=["date","price","price_1","price_2","price_3","price_4","price_5","price_6","price_7","price_8","price_9","price_10","price_11","return","return_1","return_2","return_3","return_4","return_5","return_6","return_7","return_8","return_9","return_10","return_11","range","range_1","range_2","range_3","range_4","range_5","range_6","range_7","range_8","range_9","range_10","range_11","volume","volume_1","volume_2","volume_3","volume_4","volume_5","volume_6","volume_7","volume_8","volume_9","volume_10","volume_11","sales","assets","profit_margin","asset_turnover","financial_leverage","roe","price_to_earnings","price_to_book","price_to_sales","ind_agric","ind_mines","ind_oil","ind_stone","ind_cnstr","ind_food","ind_smoke","ind_txtls","ind_apprl","ind_wood","ind_chair","ind_paper","ind_print","ind_chems","ind_ptrlm","ind_rubbr","ind_lethr","ind_glass","ind_metal","ind_mtlpr","ind_machn","ind_elctr","ind_cars","ind_instr","ind_manuf","ind_trans","ind_phone","ind_tv","ind_utils","ind_garbg","ind_steam","ind_water","ind_whlsl","ind_rtail","ind_money","ind_srvc","ind_govt","ind_other"])

  #from organize_data import read_data 
  # Shuffle data
  #n = training_data.shape[0]
  #index = np.arange(n)
  #np.random.shuffle(index)
  #training_y = training_y[index] #matches x & its y
  #training_x = training_x.values[index]
  
  #Split into train and validation sets
  testing_x = training_x[training_x['date'] > "2012-01-01"]
  testing_y = training_y[training_y['date'] > "2012-01-01"]
  testing_x = testing_x.drop('date', axis=1)
  testing_y = testing_y.drop('date', axis=1)

  training_x = training_x[training_x['date']<"2012-01-01"]
  training_y = training_y[training_y['date']<"2012-01-01"]
  training_x = training_x.drop('date', axis=1).values.astype("float32")
  training_y = training_y.drop('date', axis=1).values.astype("float32")

  training = TrainingDataSet(training_x, training_y)

  return training 


