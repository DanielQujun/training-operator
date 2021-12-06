import tensorflow as tf
import matplotlib.pyplot as plt
import os
import shutil
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.saved_model import tag_constants

flags=tf.app.flags
flags.DEFINE_string("data_dir", "/tmp/mnist-data",
                    "Directory for storing mnist data")
flags.DEFINE_string("model_dir", "/tmp/model",
                    "Directory for storing mnist model")

FLAGS = flags.FLAGS

mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.test.images.shape, mnist.test.labels.shape)
print(mnist.validation.images.shape, mnist.validation.labels.shape)

sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W)+b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()


for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

model_path=os.path.join(FLAGS.model_dir,"model")
if os.path.exists(model_path):
    shutil.rmtree(model_path)
builder = tf.saved_model.builder.SavedModelBuilder(model_path)

signature = tf.saved_model.predict_signature_def(inputs={'myInput': x},
                                                 outputs={'myOutput': y})
builder.add_meta_graph_and_variables(sess=sess,
                                     tags=[tag_constants.SERVING],
                                     signature_def_map={'predict': signature})
builder.save()


print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))