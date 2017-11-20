"""Builds the deep_thought network.

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.
4. evaluation() - Evaluates the quality of the model.
"""

import tensorflow as tf
import math

# Number of features in the numerai data
N_FEATURES = 50

def inference(x, hidden1_units, hidden2_units, hidden3_units, keep_prob):
    """Build the model for predicting probabilities.

    Args:
      x: x placeholder, from inputs().
      hidden1_units: Size of the first hidden layer.
      hidden2_units: Size of the second hidden layer.
      hidden3_units: Size of the third hidden layer.
      keep_prob: The probability that each element is kept. (Used to prevent overfitting)

    Returns:
      probabilities: Output tensor with the computed probabilities
    """
    # Hidden 1
    with tf.name_scope('hidden1'):
        weights = tf.Variable(tf.random_normal([N_FEATURES, hidden1_units], stddev=tf.sqrt(2/N_FEATURES)), name = 'weights')
        logits = tf.matmul(x, weights, name = 'logits')
        mean, variance = tf.nn.moments(logits, [0], name = 'moments')
        betas = tf.Variable(tf.zeros([hidden1_units]), name = 'betas')
        normalized_logits = tf.nn.batch_normalization(logits, mean, variance, betas, None, 1e-5, name = 'batch_norm')
        relu = tf.nn.relu(normalized_logits, name = 'relu')
        dropout = tf.nn.dropout(relu, keep_prob, name = 'dropout')
        hidden1 = dropout
    # Hidden 2
    with tf.name_scope('hidden2'):
        weights = tf.Variable(tf.random_normal([hidden1_units, hidden2_units], stddev=tf.sqrt(2/hidden1_units)), name = 'weights')
        logits = tf.matmul(hidden1, weights, name = 'logits')
        mean, variance = tf.nn.moments(logits, [0], name = 'moments')
        betas = tf.Variable(tf.zeros([hidden2_units]), name = 'betas')
        normalized_logits = tf.nn.batch_normalization(logits, mean, variance, betas, None, 1e-5, name = 'batch_norm')
        relu = tf.nn.relu(normalized_logits, name = 'relu')
        dropout = tf.nn.dropout(relu, keep_prob, name = 'dropout')
        hidden2 = dropout
    # Hidden 3
    with tf.name_scope('hidden3'):
        weights = tf.Variable(tf.random_normal([hidden2_units, hidden3_units], stddev=tf.sqrt(2/hidden2_units)), name = 'weights')
        logits = tf.matmul(hidden2, weights, name = 'logits')
        mean, variance = tf.nn.moments(logits, [0], name = 'moments')
        betas = tf.Variable(tf.zeros([hidden3_units]), name = 'betas')
        normalized_logits = tf.nn.batch_normalization(logits, mean, variance, betas, None, 1e-5, name = 'batch_norm')
        relu = tf.nn.relu(normalized_logits, name = 'relu')
        dropout = tf.nn.dropout(relu, keep_prob, name = 'dropout')
        hidden3 = dropout
    with tf.name_scope('probabilities'):
        weights = tf.Variable(tf.random_normal([hidden3_units, 2], stddev=tf.sqrt(1/hidden3_units)), name = 'weights')
        biases = tf.Variable(tf.zeros([2]), name = 'bias')
        logits = tf.matmul(hidden3, weights) + biases
        probabilities = tf.nn.softmax(logits, name = 'probabilities')
    return probabilities

def loss(predictions, y):
    """Calculates the loss function using the log loss model

    Args:
      probabilities: Tensor of probabilities from deep_thought()
      target: Tensor of target values, int32 - [batch_size]

    Returns:
      loss: loss tensor of type float.
    """
    y = tf.to_int32(y)
    loss = tf.losses.log_loss(labels=y, predictions=predictions)
    return tf.reduce_mean(loss, name='loss')

def training(loss, learning_rate):
    """Sets up the training Ops.

    Creates a summarizer to track the losses over time in TensorBoard.

    Creates an optimizer and applies the gradients to all trainable variables.

    The Op returned by this function is what must be passed to the
    'sess.run()' call to cause the model to train.

    Args:
      loss: Loss tensor from loss()
      learning_rate: The learning rate to be used for the optimizer.

    Returns:
      train_op: The operation for training.
    """
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)
    # Create the Adam optimizer with the given learning rate
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Create a variable to track the global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradiants that minimize the loss
    # and increment the global step counter as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op

def evaluation(probabilities, y):
    """Evaluate the quality of the model at predicting the target.

    Args:
      probabilities: Probabilities tensor
      target: Target tensor

    Returns:
      A scalar int32 tensor with the number of examples that were
      predicted correctly.
    """
    correct = tf.equal(tf.argmax(y, 1), tf.argmax(probabilities, 1))
    return tf.reduce_sum(tf.cast(correct, tf.int32))
