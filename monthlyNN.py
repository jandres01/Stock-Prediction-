import tensorflow as tf
import numpy as np
import monthly_organize_data as mod
import pandas as pd

def generateData():
    mnist = mod.read_data()
    mnist[['price_max','return_max','volume_max','range_max']]= mnist.groupby('permno')['price','return','volume','range'].transform('max')
    mnist['price_norm'] = mnist['price']/(mnist['price_max'] * 1.25)
    mnist['return_norm'] = mnist['return']/(mnist['return_max'] * 1.25)
    mnist['volume_norm'] = mnist['volume']/(mnist['volume_max'] * 1.25)
    mnist['range_norm'] = mnist['range']/(mnist['range_max'] * 1.25)
    mnist['permnoCount'] = mnist.groupby(['permno'])['permno'].transform('count')
    mnist = mnist[(mnist.permnoCount == 756)]
    dataY = mnist.pivot(index='permno',columns='date',values='price_norm')
    dataX = np.empty((2221,756,4))
    dataX[:,:,0] = pd.pivot_table(mnist,values='price_norm',index='permno',columns='date')
    dataX[:,:,1] = pd.pivot_table(mnist,values='return_norm',index='permno',columns='date')
    dataX[:,:,2] = pd.pivot_table(mnist,values='volume_norm',index='permno',columns='date')
    dataX[:,:,3] = pd.pivot_table(mnist,values='range_norm',index='permno',columns='date')
    y = dataY.values
    x = dataX
    return TrainingDataSet(x,y)
#hidden layer for complex

company_size = 2221
dates = 756
mnist = mod.generateData()

# Multiplying 3D matrices
## tf.random_normal - outputs random values from normal distribution
## tf.variable - maintains state in the graph across calls to run()
X = tf.placeholder(tf.float32,[None,dates,4],name="X")
W1 = tf.Variable(tf.random_normal([100,4,1], stddev = 0.001))
b1 = tf.Variable(tf.zeros([4]))

#H1 to Y
#softmax is my squashing function or normalized exponential function 
Y = tf.nn.softmax(tf.matmul(X, W1) + b1)[:,:,0] 

Y_ = tf.placeholder(tf.float32,[None,dates],name="Ylabel")

#Changed loss function from cross entropy to reduce sum of diff(actual-prediction)
#New Loss/Cost function
#loss = tf.reduce_sum(tf.abs(Y-Y_))
loss = tf.reduce_mean(tf.squared_difference(Y,Y_))
#compute mean elements across dimension of a tensor while casting new type
accuracy = tf.reduce_mean(tf.abs(Y-Y_)) 

#implement optimizer but test for other kinds as well
#Optimization does calculation of direction which weights & bias have to be changed during training so we can minimize the cost function (how the NN learns)
train_step = tf.train.AdamOptimizer().minimize(loss)

#operation that initializes all global variables 
init = tf.global_variables_initializer()
#sigma = 1
#init = tf.variance_scaling_initializer(mode="fan_avg",distribution="uniform", scale="sigma")
#bias_initializer = tf.zeros_initializer()
sess = tf.Session()
sess.run(init)

#plotting predictions of first 5 companies 
def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)
    for batch_series_idx in range(5):
        #one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        #single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])
        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        #plt.bar(left_offset, batchX[batch_series_idx, :,:], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :], width=1, color="red")
        #plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")
    plt.draw()
    plt.pause(0.0001)

#one full sweep of all the batches is an epoch
current_epoch = 1
mnist._epochs_completed=0
truncated_backprop_length = 756

#setup plotting
plt.ion()
fig = plt.figure()
ax1 = fig.add_subplot(111)
line1, = ax1.plot(Y)
line2, = ax1.plot(Y*0.5)
plt.show()

for i in range(0,10000):
    batch_X,batch_Y = mnist.next_batch(100)
    train_data = {X:batch_X, Y_:batch_Y}
    _,y = sess.run([train_step,Y],feed_dict=train_data)
    #print mnist.epochs_completed
    if current_epoch == mnist.epochs_completed:
        a,c=sess.run([accuracy,loss],feed_dict={X: batch_X, Y_: batch_Y})
        print "Epoch: ", current_epoch, "     Loss: ", c, "     Accuracy: ", a
        current_epoch += 1
            # Prediction
        #plt.title('Epoch ' + str(current_epoch) )
        #file_name = 'img/epoch_' + str(current_epoch) + '.jpg'
        #plt.savefig(file_name)
        #plt.pause(0.01)
        #plot(c,a,batch_X,batch_Y)

plt.ioff()
# Print final MSE after Traininmse_final g
#= net.run(loss, feed_dict={X: mnist._x, Y: mnist._y})
#print(mse_final)


#Name your placeholders so you can identify if there are placeholders not being initialized
