import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from sqlite3 import Error
from sklearn.preprocessing import MinMaxScaler

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
    database = '/home/jandres/data/revised-monthly-momentum-financial-data.sqlite'
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
      avgReturns[date] = dbd.get_group(date).query('total_past_return_perc >= 0.8')['next_month_return'].mean()
      avgReturns[date] -= dbd.get_group(date).query('total_past_return_perc <= 0.2')['next_month_return'].mean()
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
      c1 =sess.run([loss],feed_dict={X: x_train, Y_: y_train})
      c2 =sess.run([loss],feed_dict={X: x_test, Y_: y_test})
      print "Epoch: ",j,"\t Train Loss: ", c1, "\t Test Loss: ",c2

def splitData(testing,training):
    #Split data testing & training
    x_train = training[['return_1','return_2','return_3','return_4','return_5','return_6','return_7','return_8','return_9','return_10','return_11','total_return','perc_1','perc_2','perc_3','perc_4','perc_5','perc_6','perc_7','perc_8','perc_9','perc_10','perc_11','prod_1','prod_2','prod_3','prod_4','prod_5','prod_6','prod_7','prod_8','prod_9','prod_10','prod_11','total_return_prod','total_return_perc']]
    y_train = training[['next_month_return_perc']]
    x_test = testing[['return_1','return_2','return_3','return_4','return_5','return_6','return_7','return_8','return_9','return_10','return_11','total_return','perc_1','perc_2','perc_3','perc_4','perc_5','perc_6','perc_7','perc_8','perc_9','perc_10','perc_11','prod_1','prod_2','prod_3','prod_4','prod_5','prod_6','prod_7','prod_8','prod_9','prod_10','prod_11','total_return_prod','total_return_perc']]
    y_test = testing[['next_month_return_perc']]
    dbd = testing.groupby('date')
    all_dates = []
    for group in dbd.groups:
      all_dates.append(group)
    output = pd.DataFrame(columns = ["date"])
    output["date"] = all_dates
    return testing,training,x_train,y_train,x_test,y_test,output

def evenlySplitData():
    a,b,c,d,e = np.array_split(data,5)
    a_date = a.iloc[0]['date'] 
    b_date = b.iloc[0]['date'] 
    c_date = c.iloc[0]['date'] 
    d_date = d.iloc[0]['date'] 
    e_date = e.iloc[0]['date'] 
    return a_date,b_date,c_date,d_date,e_date

data = read_data()
data = data.sort(['date','permno'])
date = "2011-11-20"
testing = data[data['date'] >= date] #testing
training = data[data['date']< date]

aDate,bDate,cDate,dDate,eDate = evenlySplitData()
testing_e = data[data['date'] >= eDate]
training = data[data['date'] < eDate]

test_data,train_data,x_train,y_train,x_test,y_test,output = splitData(testing,training)

epochs = 100
n_passed_values = 36
n_neurons_1 = 700 #512 
n_neurons_2 = 350 #256
n_neurons_3 = 175 #128
n_target = 1

X = tf.placeholder(tf.float32,[None,n_passed_values],name="X")

Y_ = tf.placeholder(tf.float32,[None,1],name="Ylabel")

# Initializers
weight_initializer = tf.contrib.layers.variance_scaling_initializer(mode='FAN_AVG', uniform=True, factor=1.0, seed=None)
bias_initializer = tf.zeros_initializer()

#Variables for Hidden layers: [Input,Output]
W_hidden_1 = tf.Variable(weight_initializer([n_passed_values, n_neurons_1]))
bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))

W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))

W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))

# Output layer: Variables for output weights and biases
W_out = tf.Variable(weight_initializer([n_neurons_3, n_target]))
bias_out = tf.Variable(bias_initializer([n_target]))

# Hidden layer
#test Relu - rectified linear unit - helps decrease vanishing gradient
hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))

#feedforward network
Y = tf.add(tf.matmul(hidden_3, W_out), bias_out)

loss = tf.reduce_mean(tf.squared_difference(Y,Y_))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#Calculate the average return of momentum theory portfolio
#calcMomentumReturns(test_data)

#Call to calculate loss function of the NN
#calcLossNN()

#NN_Pred predicts next_month_return_percentile
def calcNN_Pred(testData,xTest):
    calcLossNN()
    #Create predictions with unshuffled test data
    y = sess.run(Y, feed_dict={X:xTest.values})
    #print y
    predData = testData.assign(nn_pred= y)
    dbd = predData.groupby('date')
    all_dates = []
    for group in dbd.groups:
      all_dates.append(group)
    #Select Portfolio
    nAvgReturns = {}
    results = []
    for date in all_dates:
      nAvgReturns[date] = dbd.get_group(date).nlargest(int(dbd.get_group(date)['permno'].nunique() *0.2),'nn_pred')['next_month_return'].mean()
      nAvgReturns[date] -= dbd.get_group(date).nsmallest(int(dbd.get_group(date)['permno'].nunique() *0.2),'nn_pred')['next_month_return'].mean()
      results.append(nAvgReturns[date])
    #Print Portfolio return
#    for date in sorted(nAvgReturns.keys()):
#      print(str(date) + "\t" + str(nAvgReturns[date]))
    return results

loop = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10"]

#loop = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10","c11","c12","c13","c14","c15","c16","c17","c18","c19","c20","c21","c22","c23","c24","c25","c26","c27","c28","c29","c30","c31","c32","c33","c34","c35","c36","c37","c38","c39""c40","c41","c42","c43","c44","c45","c46","c47","c48","c49","c50","c51","c52","c53","c54","c55","c56","c57","c58","c59","c60","c61","c62","c63","c64","c65","c66","c67","c68","c69","c70","c71","c72","c73","c74","c75","c76","c77","c78","c79","c80","c81","c82","c83","c84","c85","c86","c87","c88","c89""c90","c91","c92","c93","c94","c95","c96","c97","c98","c99","c100"]

for i in loop:
    output[i] = calcNN_Pred(test_data,x_test)
 
#Calculate average return of NN portfolio
#calcNN_Pred()
output.to_csv("/home/jandres/data/nmrp_5_10_momentum_avgReturns.csv")


