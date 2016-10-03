import tensorflow as tf
import random

x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])

W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))
y = tf.tanh(tf.matmul(x, W) + b)

loss = tf.reduce_mean(tf.abs(y - y_))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()

session = tf.Session()
session.run(init)


for step in range(0, 100):
    x_datas = []
    y_datas = []
    for batch in range(0, 10):
        a, b = random.randint(0, 1), random.randint(0, 1)
        x_datas.append([a, b])
        y_datas.append([a | b])
    session.run(train, feed_dict={x: x_datas, y_: y_datas})

#evaluation
for a in range(0, 2):
    for b in range(0, 2):
        print a, b
        print session.run(y, feed_dict={x: [[a, b]]})
        print a | b
        print

