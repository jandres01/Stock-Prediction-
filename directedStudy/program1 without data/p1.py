import tensorflow as tf
import numpy as np
import organize_data as od

mnist = od.read_data("./data/return-data.csv")

#hidden layer for complex

X = tf.placeholder(tf.float32,[None,95])
W1 = tf.Variable(tf.random_normal([95,200], stddev = 0.001))
b1 = tf.Variable(tf.zeros([200]))
W2 = tf.Variable(tf.random_normal([200,100], stddev = 0.001))
b2 = tf.Variable(tf.zeros([100]))
W3 = tf.Variable(tf.random_normal([100,10], stddev = 0.001))
b3 = tf.Variable(tf.zeros([10]))
 
H1 = tf.nn.softmax(tf.matmul(X, W1) + b1) 
H2 = tf.nn.softmax(tf.matmul(H1, W2) + b2)
Y = tf.nn.softmax(tf.matmul(H2, W3) + b3)

Y_ = tf.placeholder(tf.float32,[None,10])

is_correct = tf.equal(tf.argmax(Y,1),tf.argmax(Y_,1))

cross_entropy = tf.losses.log_loss(labels = Y_, predictions = Y)

accuracy = tf.reduce_mean(tf.cast(is_correct,tf.float32))

optimizer = tf.train.AdamOptimizer(0.0003)

train_step = optimizer.minimize(cross_entropy)

#adam optimzer intialization happens after declaration
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

current_epoch = 1
for i in range(0,10000000):
    batch_X,batch_Y = mnist.next_batch(100)
    train_data = {X:batch_X, Y_:batch_Y}
    _,y = sess.run([train_step,Y],feed_dict=train_data)
    #print y
    if current_epoch == mnist.epochs_completed:
        a,c=sess.run([accuracy,cross_entropy],feed_dict={X: mnist.x, Y_: mnist.y})
        print "Epoch: ", current_epoch, "     Loss: ", c, "     Accuracy: ", a
        current_epoch += 1
    #test_data={X:mnist.test.images, Y_:mnist.test.labels}
    #a,c = sess.run([accuracy,cross_entropy],feed_dict=test_data)



