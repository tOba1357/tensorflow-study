import tensorflow as tf
import random

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 2])

W = tf.Variable(tf.zeros([2, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cross_entropy)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)


for step in range(0, 1000):
    ab = [random.randint(0, 1), random.randint(0, 1)]
    ans = [1, 0] if ab[0] | ab[1] == 0 else [0, 1]
    session.run(train, feed_dict={x: [ab], y_: [ans]})

#evaluation
for a in range(0, 2):
    for b in range(0, 2):
        print a, b
        print session.run(y, feed_dict={x: [[a, b]]})
        print a | b
        print

