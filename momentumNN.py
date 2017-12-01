import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
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
    #show tables in db
    res = cur.execute("Select name from sqlite_master WHERE type='table';")
    for name in res:
        print name[0]
    #Show column names
    data = cur.execute("SELECT * FROM crsp LIMIT 10")
    print(list(map(lambda x: x[0], data.description)))
    #Convert date to Unix for filtering in query for testing
    #s = "2012-01-01"
    #time.mktime(datetime.datetime.strptime(s,"%Y-%m-%d").timetuple())
    #grab Data
    dfData = pd.read_sql_query("select * from crsp",conn)
    #drop all rows with NaN & convert date
    dfData.dropna(axis = 0, how='any')
    dfData['date'] = pd.to_datetime(dfData['date'],unit='d')
    return dfData

def read_data():
    database = '/home/jandres/data/momentum-financial-data.sqlite'
    conn = create_connection(database)
    data = select_all_tasks(conn)
    return data

def calcMomentumReturns(data):
    #data = read_data()
    dbd = data.groupby('date')
    all_dates = []
    for group in dbd.groups:
      all_dates.append(group)
    
    avgReturns = {}
    for date in all_dates:
      avgReturns[date] = dbd.get_group(date).query('total_past_return_percentile >= 0.8')['next_month_return'].mean()
      avgReturns[date] -= dbd.get_group(date).query('total_past_return_percentile <= 0.2')['next_month_return'].mean()
    
    for date in sorted(avgReturns.keys()):
      print(str(date) + "\t" + str(avgReturns[date]))

def calcLossNN(x_data,y_data):
    #First shuffle data
    batch_size = 10
    shuffle_indices = np.random.permutation(np.arange(len(y_data)))
    X_train = x_data.values[shuffle_indices]
    y_train = y_data.values[shuffle_indices]
    #predict loss by grabbing batches
    #epoch is finished cycle of grabbing all batches
    for i in range(0, len(y_train) // batch_size):
      start = i * batch_size
      batch_x = X_train[start:start + batch_size]
      batch_y = y_train[start:start + batch_size]
      train_data = {X:batch_x,Y_:batch_y}
      _,y = sess.run([train_step,Y],feed_dict=train_data)
      c=sess.run([loss],feed_dict={X: batch_x, Y_: batch_y})
      print "Epoch: ", i , "     Loss: ", c


data = read_data()
n = data.shape[0]

def parseData():
    #Split data testing & training
    train_start = 0
    train_end = int(np.floor(0.8*n))
    test_start = train_end + 1
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]
    return data_train,data_test

#x_whole_data,y_whole_data = parseData()

cDate = data[['permno','date']]
x_data = data[['return_lag_1','return_lag_2','return_lag_3','return_lag_4','return_lag_5','return_lag_6','return_lag_7','return_lag_8','return_lag_9','return_lag_10','return_lag_11']]
y_data = data[['next_month_return_percentile']]



X = tf.placeholder(tf.float32,[None,11],name="X")
W1 = tf.Variable(tf.random_normal([11,1], stddev = 0.001))
b1 = tf.Variable(tf.zeros([1]))

Y = tf.matmul(X, W1) + b1
Y_ = tf.placeholder(tf.float32,[None,1],name="Ylabel")

loss = tf.reduce_mean(tf.squared_difference(Y,Y_))
train_step = tf.train.AdamOptimizer().minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Calculate the average return of momentum theory portfolio
#calcMomentumReturns(data)

#Call to calculate loss function of the NN
#calcLossNN(x_data,y_data)

#NN_Pred predicts next_month_return_percentile
def calcNN_Pred():
    #Predictions
    y = sess.run(Y, feed_dict={X:x_data.values, Y_:y_data.values})
    #print y
    predData = data.assign(nn_pred= y)
    dbd = predData.groupby('date')
    all_dates = []
    for group in dbd.groups:
      all_dates.append(group)
    # Returns NaN for everything
    #avgReturns = {}
    #for date in all_dates:
    #    avgReturns[date] = dbd.get_group(date).query("nn_pred >= 0.8")['next_month_return'].mean()
    #    avgReturns[date] -= dbd.get_group(date).query("nn_pred <= 0.2")['next_month_return'].mean()
    #for date in sorted(avgReturns.keys()):
    #    print(str(date) + "\t" + str(avgReturns[date])) 
    nAvgReturns = {}
    for date in all_dates:
      nAvgReturns[date] = dbd.get_group(date).nlargest(int(data['permno'].nunique() *0.2),'nn_pred')['next_month_return'].mean()
      nAvgReturns[date] -= dbd.get_group(date).nsmallest(int(data['permno'].nunique() *0.2),'nn_pred')['next_month_return'].mean()
    for date in sorted(nAvgReturns.keys()):
      print(str(date) + "\t" + str(nAvgReturns[date]))

#Calculate average return of NN portfolio
#calcNN_Pred()




