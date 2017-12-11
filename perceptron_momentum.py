import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from sqlite3 import Error


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

def calcLossNN():
    #First shuffle data
    batch_size = 250
    for j in range(0, epochs):
      shuffle_indices = np.random.permutation(np.arange(len(y_train)))
      X_train = x_train.values[shuffle_indices]
      Y_train = y_train.values[shuffle_indices]
    #predict loss by grabbing batches
    #epoch is finished cycle of grabbing all batches
      for i in range(0, len(y_train) // batch_size):
        start = i * batch_size
        batch_x = X_train[start:start + batch_size]
        batch_y = Y_train[start:start + batch_size]
        train_data = {X:batch_x,Y_:batch_y}
        _,y = sess.run([train_step,Y],feed_dict=train_data)
      c=sess.run([loss],feed_dict={X: x_train, Y_: y_train})
      print "Epoch: ",j,"\t Loss: ", c

def splitData():
    #Split data testing & training
    testing = data[data['date'] >= "2011-11-30"] #testing
    training = data[data['date']< "2011-11-30"]
    x_train = training[['return_lag_1','return_lag_2','return_lag_3','return_lag_4','return_lag_5','return_lag_6','return_lag_7','return_lag_8','return_lag_9','return_lag_10','return_lag_11','total_past_return_percentile']]
    y_train = training[['next_month_return']]
    x_test = testing[['return_lag_1','return_lag_2','return_lag_3','return_lag_4','return_lag_5','return_lag_6','return_lag_7','return_lag_8','return_lag_9','return_lag_10','return_lag_11','total_past_return_percentile']]
    y_test = testing[['next_month_return']]
    return testing,training,x_train,y_train,x_test,y_test

data = read_data()
test_data,train_data,x_train,y_train,x_test,y_test = splitData()

epochs = 100

X = tf.placeholder(tf.float32,[None,12],name="X")
W1 = tf.Variable(tf.random_normal([12,1], stddev = 0.001))
b1 = tf.Variable(tf.zeros([1]))

Y = tf.matmul(X, W1) + b1
Y_ = tf.placeholder(tf.float32,[None,1],name="Ylabel")

loss = tf.reduce_mean(tf.squared_difference(Y,Y_))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Calculate return of momentum theory portfolio
#calcMomentumReturns(data)

#Call to calculate loss function of the NN
calcLossNN()

#NN_Pred predicts next_month_return_percentile
def calcNN_Pred():
    #Create predictions with unshuffled test data
    _,y = sess.run([train_step,Y], feed_dict={X:x_test.values, Y_:y_test.values})
    #print y
    predData = test_data.assign(nn_pred= y)
    dbd = predData.groupby('date')
    all_dates = []
    for group in dbd.groups:
      all_dates.append(group)
    #Select Portfolio
    nAvgReturns = {}
    for date in all_dates:
      nAvgReturns[date] = dbd.get_group(date).nlargest(int(dbd.get_group(date)['permno'].nunique() *0.2),'nn_pred')['next_month_return'].mean()
      nAvgReturns[date] -= dbd.get_group(date).nsmallest(int(dbd.get_group(date)['permno'].nunique() *0.2),'nn_pred')['next_month_return'].mean()
    #Print Portfolio return
    for date in sorted(nAvgReturns.keys()):
      print(str(date) + "\t" + str(nAvgReturns[date]))

#Calculate return of NN portfolio for the month
calcNN_Pred()




