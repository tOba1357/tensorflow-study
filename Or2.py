import tensorflow as tf
import random

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.tanh(tf.matmul(x, W) + b)

mean_squared_error = tf.reduce_mean(tf.square(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(mean_squared_error)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)

def evaluation():
    for a in range(0, 2):
        for b in range(0, 2):
            print "a:%d, b:%d" % (a, b)
            print session.run(y, feed_dict={x: [[a, b]]})[0][0]

for step in range(0, 100):
    x_data = [random.randint(0, 1), random.randint(0, 1)]
    y_data = [x_data[0] | x_data[1]]
    session.run(train, feed_dict={x: [x_data], y_: [y_data]})
    if step % 10 == 0:
        print "step:%d" % (step)
        evaluation()
        print

evaluation()