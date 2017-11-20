#!/usr/bin/env python

"""
Example classifier on Numerai data using a logistic regression classifier.
To get started, install the required packages: pip install pandas, numpy, sklearn
"""

import pandas as pd
import numpy as np
import deep_thought
import numerai_data
import tensorflow as tf
import os.path
import math

BATCH_SIZE = 500
HIDDEN1_UNITS = 38
HIDDEN2_UNITS = 26
HIDDEN3_UNITS = 14
KEEP_PROB = 0.5
MAX_STEPS = 5000000
LEARNING_RATE = 0.0004

# Get the numerai datasets loaded
training, testing, tournament = numerai_data.read_data_sets("data/numerai_training_data.csv", "data/numerai_tournament_data.csv")

def do_eval(sess,
            eval_correct,
            X,
            Y,
            data_set):
  """
  Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    X: The X placeholder.
    Y: The Y placeholder.
    data_set: The dataset to evaluate, from numerai_data.read_data_sets().
  """
  # And run one epoch of eval.
  number_correct = sess.run(eval_correct, {X: data_set.x, Y: data_set.y})
  accuracy = float(number_correct) / data_set.n
  return accuracy

def run_training():
  """
  Train deep_thought for a number of steps.
  """

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Generate placeholders for the images and labels.
    X = tf.placeholder(tf.float32, shape=(None, deep_thought.N_FEATURES))
    Y = tf.placeholder(tf.int32, shape=(None, 2))
    LR = tf.placeholder(tf.float64)

    # Build a Graph that computes predictions from the inference model.
    predictions = deep_thought.inference(X, HIDDEN1_UNITS, HIDDEN2_UNITS, HIDDEN3_UNITS, KEEP_PROB)

    # Add to the Graph the Ops for loss calculation.
    loss = deep_thought.loss(predictions, Y)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = deep_thought.training(loss, LR)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = deep_thought.evaluation(predictions, Y)

    # Build the summary Tensor based on the TF collection of Summaries.
    summary = tf.summary.merge_all()

    # Add the variable initializer Op.
    init = tf.global_variables_initializer()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # And then after everything is built:
    # Run the Op to initialize the variables.
    sess.run(init)

    current_epoch = 1
    epochs_since_new_best = 0
    best_testing_loss_moving_avg = 1.0
    best_testing_accuracy_moving_avg = 0.0
    testing_loss_moving_avg = 1.0
    testing_accuracy_moving_avg = 1.0
    max_smoothing_factor = 1.0
    min_smoothing_factor = 0.01
    decay_speed = 15.0
    max_prediction_smoothing_factor = 1.0
    min_prediction_smoothing_factor = 0.1
    prediction_decay_speed = 20.0
    # Start the training loop.
    for step in range(MAX_STEPS):
      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      X_feed, Y_feed = training.next_batch(BATCH_SIZE)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      sess.run(train_op, {LR: LEARNING_RATE, X: X_feed, Y: Y_feed})

      # Write the summaries and print an overview fairly often.
      if current_epoch == training.epochs_completed:
        # Print status to stdout.
        training_loss = sess.run(loss, {X: training.x, Y: training.y})
        training_accuracy = do_eval(sess, eval_correct, X, Y, training)
        testing_loss = sess.run(loss, {X: testing.x, Y: testing.y})
        testing_accuracy = do_eval(sess, eval_correct, X, Y, testing)
        smoothing_factor = min_smoothing_factor + (max_smoothing_factor - min_smoothing_factor) * math.exp(-(current_epoch - 1) / decay_speed)
        testing_loss_moving_avg = (smoothing_factor * testing_loss) + ((1 -  smoothing_factor) * testing_loss_moving_avg)
        testing_accuracy_moving_avg = (smoothing_factor * testing_accuracy) + ((1 - smoothing_factor) * testing_accuracy_moving_avg)
        print('Completed Epoch %d:  Training Accuracy: %.5f   Loss: %.5f   Testing Accuracy: %.5f   Loss: %.5f   Avg Testing Accuracy:   %.5f   Loss:  %.5f' % (current_epoch, training_accuracy, training_loss, testing_accuracy, testing_loss, testing_accuracy_moving_avg, testing_loss_moving_avg))
        if testing_loss_moving_avg < best_testing_loss_moving_avg:
            best_testing_loss_moving_avg = testing_loss_moving_avg
        if testing_accuracy_moving_avg > best_testing_accuracy_moving_avg:
            best_testing_accuracy_moving_avg = testing_accuracy_moving_avg
        if testing_loss_moving_avg == best_testing_loss_moving_avg or testing_accuracy_moving_avg == best_testing_accuracy_moving_avg:
            current_predictions = sess.run(predictions, feed_dict={X: tournament.x})
            if current_epoch == 1:
                updated_predictions = current_predictions
            else:
                prediction_smoothing_factor = min_prediction_smoothing_factor + (max_prediction_smoothing_factor - min_prediction_smoothing_factor) * math.exp(-(current_epoch - 1) / prediction_decay_speed)
                updated_predictions = (prediction_smoothing_factor * current_predictions) + ((1 - prediction_smoothing_factor) * updated_predictions)
            epochs_since_new_best = 0
        else:
            epochs_since_new_best += 1
        if epochs_since_new_best >= 10:
            break
        current_epoch += 1
    tournament_predictions = pd.DataFrame(updated_predictions[:,1], columns = ['probability'])
    results = pd.DataFrame(tournament.t_id, columns = ['t_id']).join(tournament_predictions)
    print('Predictions Summary...')
    print(tournament_predictions.describe(percentiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]))
    print('Writing predictions to predictions.csv...')
    results.to_csv('data/predictions.csv', index=False)


def main():
  run_training()

if __name__ == '__main__':
    main()
