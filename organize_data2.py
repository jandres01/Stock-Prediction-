import numpy as np
import pandas as pd
import sqlite3
from sqlite3 import Error

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

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)

    return None

def select_all_tasks(conn):
    cur = conn.cursor()
    data = cur.execute("SELECT * FROM crsp LIMIT 10")
    print(list(map(lambda x: x[0], data.description))) 
    dfData = pd.read_sql_query("select * from crsp",conn)
    #Before below drop all rows with NaN & those with lack of days
    dfData.dropna(axis = 0, how='any')
    dfData['date'] = pd.to_datetime(dfData['date'],unit='d')
    #dfData[['price_max','return_max','volume_max']]= dfData.groupby('gvkey')['price','return','volume'].transform('max')
    #dfData['price_norm'] = dfData['price']/(dfData['price_max'] * 1.25)
    #dfData['return_norm'] = dfData['return']/(dfData['return_max'] * 1.25)
    #dfData['volume_norm'] = dfData['volume']/(dfData['volume_max'] * 1.25)
    #dfData['gvkeyCount'] = dfData.groupby(['gvkey'])['gvkey'].transform('count')
    #dfData = dfData[(dfData.gvkeyCount == 756)]
    return dfData

def read_data():
    database = '/home/jandres/data/financial-data.sqlite'
    conn = create_connection(database)
    data = select_all_tasks(conn)
  
    #Split into train and validation sets
    testing = data[data['date'] > "2014-01-01"]
    training = data[data['date']<= "2014-01-01"]

    return testing


