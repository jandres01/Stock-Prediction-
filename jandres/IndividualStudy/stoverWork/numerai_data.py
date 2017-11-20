import numpy as np
import pandas as pd

class TrainingDataSet(object):

  def __init__(self, x, y):
    """
    Construct a DataSet for training data.
    """
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
    """
    Return the next `batch_size` worth of data from this data set.
    """
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

class TournamentDataSet(object):

  def __init__(self, x, t_id):
    """
    Construct a DataSet for holding tournament data.
    """
    assert x.shape[0] == t_id.shape[0], (
          'size of x: %s size of t_id: %s' % (x.shape[0], t_id.shape[0]))

    self._n = x.shape[0]
    self._x = x
    self._t_id = t_id

  @property
  def x(self):
    return self._x

  @property
  def t_id(self):
    return self._t_id

  @property
  def n(self):
    return self._n

def dense_to_one_hot(labels_dense, num_classes):
  """
  Convert class labels from scalars to one-hot vectors.
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.astype(int).ravel()] = 1
  return labels_one_hot

def read_data_sets(training_data_file, tournament_data_file):

    training_data = pd.read_csv(training_data_file)
    training_y = training_data['target'].values.reshape(-1, 1)
    training_y = dense_to_one_hot(training_y, 2)
    training_x = training_data.drop('target', axis=1)

    # Shuffle data
    n = training_data.shape[0]
    index = np.arange(n)
    np.random.shuffle(index)
    training_y = training_y[index]
    training_x = training_x.values[index]
    # Split into train and validation sets
    testing_size = round(n/10.0) # 10% of total
    testing_x = training_x[:testing_size]
    testing_y = training_y[:testing_size]
    training_x = training_x[testing_size:]
    training_y = training_y[testing_size:]

    tournament_data = pd.read_csv(tournament_data_file)
    t_id = tournament_data['t_id'].values
    tournament_x = tournament_data.drop('t_id', axis=1)

    training = TrainingDataSet(training_x, training_y)
    testing = TrainingDataSet(testing_x, testing_y)
    tournament = TournamentDataSet(tournament_x, t_id)

    return training, testing, tournament
