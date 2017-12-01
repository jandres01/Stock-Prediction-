from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import organize_data2 as od
import pandas as pd

num_epochs = 100
total_series_length = 1000000
truncated_backprop_length = 30
state_size = 64
num_classes = 2
echo_step = 3
batch_size = 21
num_batches = total_series_length//batch_size//truncated_backprop_length
num_layers = 3

def generateData():
    #x = np.array(nddp.random.choice(2, total_series_length, p=[0.5, 0.5]))
    mnist = od.read_data()

    #mnist.dropna(axis = 0, how='any')
    #mnist['date'] = pd.to_datetime(mnist['date'],unit='d')
    mnist[['price_max','return_max','volume_max']]= mnist.groupby('gvkey')['price','return','volume'].transform('max')
    mnist['price_norm'] = mnist['price']/(mnist['price_max'] * 1.25)
    mnist['return_norm'] = mnist['return']/(mnist['return_max'] * 1.25)
    mnist['volume_norm'] = mnist['volume']/(mnist['volume_max'] * 1.25)
    mnist['gvkeyCount'] = mnist.groupby(['gvkey'])['gvkey'].transform('count')
    mnist = mnist[(mnist.gvkeyCount == 756)]
    dataY = mnist.pivot(index='gvkey',columns='date',values='price_norm')
    dataY = dataY.dropna()
    dataX = np.empty((2205,756,3))
    dataX[:,:,0] = pd.pivot_table(mnist,values='price_norm',index='gvkey',columns='date')
    dataX[:,:,1] = pd.pivot_table(mnist,values='return_norm',index='gvkey',columns='date')
    dataX[:,:,2] = pd.pivot_table(mnist,values='volume_norm',index='gvkey',columns='date')
    
    #y = np.roll(x, echo_step)
    #y[0:echo_step] = 0
    
    x = dataX.reshape((batch_size, -1,3))  # The first index changing slowest, subseries as rows
    y = dataY.values
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length,3])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [num_layers, 2, batch_size, state_size])

state_per_layer_list = tf.unstack(init_state, axis=0)
rnn_tuple_state = tuple(
    [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0,:,:], state_per_layer_list[idx][1,:,:])
     for idx in range(num_layers)]
)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
#inputs_series = tf.split(axis=1, num_or_size_splits=truncated_backprop_length, value=batchX_placeholder)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
#cell = tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True)
#cell = tf.contrib.rnn.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
stacked_rnn = []
for _ in range(num_layers):
    stacked_rnn.append(tf.contrib.rnn.LSTMCell(state_size, state_is_tuple=True))

cell = tf.contrib.rnn.MultiRNNCell(stacked_rnn, state_is_tuple=True)

states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, initial_state=rnn_tuple_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :,:], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    #plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()

        _current_state = np.zeros((num_layers, 2, batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx,:]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder: batchX,
                    batchY_placeholder: batchY,
                    init_state: _current_state
                })


            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Batch loss", _total_loss)
                #plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
#plt.show()
