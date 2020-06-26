## Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# ==================================================================
#
#        _____  ______          _____  __  __ ______ 
#       |  __ \|  ____|   /\   |  __ \|  \/  |  ____|
#       | |__) | |__     /  \  | |  | | \  / | |__   
#       |  _  /|  __|   / /\ \ | |  | | |\/| |  __|  
#       | | \ \| |____ / ____ \| |__| | |  | | |____ 
#       |_|  \_\______/_/    \_\_____/|_|  |_|______| 
#
#
# The origional fullyConnectedReader.py uses an mnist python module 
# which is now removed from our upstream source. 
# As a result, `from tensorflow.examples.tutorials.mnist import mnist`
# Will no longer work. To fix this, we copied the mnist.py from the public github repo.
# (https://github.com/tensorflow/tensorflow/blob/r1.15/tensorflow/examples/tutorials/mnist/mnist.py)
# and slightly modified it to correct the syntax differences.
#
# SIM: https://sim.amazon.com/issues/DLAMI-922
#
# ==================================================================

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE


def mnist_inference(images, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.

  Args:
    images: Images placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.compat.v1.name_scope('hidden1'):
    weights = tf.Variable(
        tf.random.truncated_normal(
            [IMAGE_PIXELS, hidden1_units],
            stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))), name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
  # Hidden 2
  with tf.compat.v1.name_scope('hidden2'):
    weights = tf.Variable(
        tf.random.truncated_normal(
            [hidden1_units, hidden2_units],
            stddev=1.0 / math.sqrt(float(hidden1_units))), name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.compat.v1.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.random.truncated_normal(
            [hidden2_units, NUM_CLASSES],
            stddev=1.0 / math.sqrt(float(hidden2_units))), name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits


def mnist_loss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.cast(labels, dtype=tf.int64)
  return tf.compat.v1.losses.sparse_softmax_cross_entropy(
      labels=labels, logits=logits)


def mnist_training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.compat.v1.summary.scalar('loss', loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def mnist_evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).

  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(predictions=logits, targets=labels, k=1)
  # Return the number of true entries.
  return tf.reduce_sum(input_tensor=tf.cast(correct, tf.int32))

# ==================================================================
#
#
#
#
#
#      Intentionally left blank to divide the code sections
#
#
#
#
#
# ==================================================================

"""Train and Eval the MNIST network.

This version is like fully_connected_feed.py but uses data converted
to a TFRecords file containing tf.train.Example protocol buffers.
See:
https://www.tensorflow.org/guide/reading_data#reading_from_files
for context.

YOU MUST run convert_to_records before running this (but you only need to
run it once).
"""

import argparse
import os.path
import sys
import time

# Basic model parameters as external flags.
FLAGS = None

# Constants used for dealing with the files, matches convert_to_records.
TRAIN_FILE = 'train.tfrecords'
VALIDATION_FILE = 'validation.tfrecords'


def decode(serialized_example):
  """Parses an image and label from the given `serialized_example`."""
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'image_raw': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor (whose single string has
  # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
  # [mnist.IMAGE_PIXELS].
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image.set_shape((IMAGE_PIXELS))

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['label'], tf.int32)

  return image, label


def augment(image, label):
  """Placeholder for data augmentation."""
  # OPTIONAL: Could reshape into a 28x28 image and apply distortions
  # here.  Since we are not applying any distortions in this
  # example, and the next step expects the image to be flattened
  # into a vector, we don't bother.
  return image, label


def normalize(image, label):
  """Convert `image` from [0, 255] -> [-0.5, 0.5] floats."""
  image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
  return image, label


def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).

    This function creates a one_shot_iterator, meaning that it will only iterate
    over the dataset once. On the other hand there is no special initialization
    required.
  """
  if not num_epochs:
    num_epochs = None
  filename = os.path.join(FLAGS.train_dir, TRAIN_FILE
                          if train else VALIDATION_FILE)

  with tf.name_scope('input'):
    # TFRecordDataset opens a binary file and reads one record at a time.
    # `filename` could also be a list of filenames, which will be read in order.
    dataset = tf.data.TFRecordDataset(filename)

    # The map transformation takes a function and applies it to every element
    # of the dataset.
    dataset = dataset.map(decode)
    dataset = dataset.map(augment)
    dataset = dataset.map(normalize)

    # The shuffle transformation uses a finite-sized buffer to shuffle elements
    # in memory. The parameter is the number of elements in the buffer. For
    # completely uniform shuffling, set the parameter to be the same as the
    # number of elements in the dataset.
    dataset = dataset.shuffle(1000 + 3 * batch_size)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset)
  return iterator.get_next()


def run_training():
  """Train MNIST for a number of steps."""

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    # Input images and labels.
    image_batch, label_batch = inputs(
        train=True, batch_size=FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist_inference(image_batch, FLAGS.hidden1, FLAGS.hidden2)

    # Add to the Graph the loss calculation.
    loss = mnist_loss(logits, label_batch)

    # Add to the Graph operations that train the model.
    train_op = mnist_training(loss, FLAGS.learning_rate)

    # The op for initializing the variables.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # Create a session for running operations in the Graph.
    with tf.Session() as sess:
      # Initialize the variables (the trained variables and the
      # epoch counter).
      sess.run(init_op)
      try:
        step = 0
        while True:  # Train until OutOfRangeError
          start_time = time.time()

          # Run one step of the model.  The return values are
          # the activations from the `train_op` (which is
          # discarded) and the `loss` op.  To inspect the values
          # of your ops or variables, you may include them in
          # the list passed to sess.run() and the value tensors
          # will be returned in the tuple from the call.
          _, loss_value = sess.run([train_op, loss])

          duration = time.time() - start_time

          # Print an overview fairly often.
          if step % 100 == 0:
            print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value,
                                                       duration))
          step += 1
      except tf.errors.OutOfRangeError:
        print('Done training for %d epochs, %d steps.' % (FLAGS.num_epochs,
                                                          step))


def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='Initial learning rate.')
  parser.add_argument(
      '--num_epochs',
      type=int,
      default=2,
      help='Number of epochs to run trainer.')
  parser.add_argument(
      '--hidden1',
      type=int,
      default=128,
      help='Number of units in hidden layer 1.')
  parser.add_argument(
      '--hidden2',
      type=int,
      default=32,
      help='Number of units in hidden layer 2.')
  parser.add_argument('--batch_size', type=int, default=100, help='Batch size.')
  parser.add_argument(
      '--train_dir',
      type=str,
      default='s3://dlami-dataset-test/Tensorflow',
      help='Directory with the training data.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)