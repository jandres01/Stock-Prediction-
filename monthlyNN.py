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
    return (x,y)
#hidden layer for complex

company_size = 2221
dates = 756
mnist = generateData()

# Multiplying 3D matrices
## tf.random_normal - outputs random values from normal distribution
## tf.variable - maintains state in the graph across calls to run()
X = tf.placeholder(tf.float32,[company_size,dates,4])
W1 = tf.Variable(tf.random_normal([company_size,4,dates], stddev = 0.001))
b1 = tf.Variable(tf.zeros([dates]))

W2 = tf.Variable(tf.random_normal([dates,100,4], stddev = 0.001))
b2 = tf.Variable(tf.zeros([100]))

W3 = tf.Variable(tf.random_normal([100,10,4], stddev = 0.001))
b3 = tf.Variable(tf.zeros([10]))

#H1 to Y 
Y = tf.nn.softmax(tf.matmul(X, W1) + b1) 
#H2 = tf.nn.softmax(tf.matmul(H1, W2) + b2)
#Y = tf.nn.softmax(tf.matmul(H2, W3) + b3)

Y_ = tf.placeholder(tf.float32,[company_size,dates,dates])

#tf.argmex returns index with largest value across axes of tensor
#Validate if largest value for both tensors are equal
is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,2))

#applies log loss method. Quantify accuracy or inaccuracy of a classifier 
cross_entropy = tf.losses.log_loss(labels = Y_, predictions = Y)

#compute mean elements across dimension of a tensor while casting new type
accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

#implement optimizer but test for other kinds as well
#Chosen optimization algorithm is used to train our model. Telling us when our model is fully optimized
optimizer = tf.train.AdamOptimizer(0.0003)

#Minimize loss - how well the network learns from mistakes
train_step = optimizer.minimize(cross_entropy)

#operation that initializes all global variables 
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

current_epoch = 1
for i in range(0,10):
    batch_X,batch_Y = mnist
    train_data = {X:batch_X, Y_:batch_Y}
    _,y = sess.run([train_step,Y],feed_dict=train_data)
    #print y
    if current_epoch == mnist.epochs_completed:
        a,c=sess.run([accuracy,cross_entropy],feed_dict={X: mnist.x, Y_: mnist.y})
        print "Epoch: ", current_epoch, "     Loss: ", c, "     Accuracy: ", a
        current_epoch += 1
    test_data={X:mnist.test.images, Y_:mnist.test.labels}
    a,c = sess.run([accuracy,cross_entropy],feed_dict=test_data)



