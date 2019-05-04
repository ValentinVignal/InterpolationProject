import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa as lb


path1 = '../Data/data_1/part_1.wav'
path2 = '../Data/data_1/part_2.wav'
y1, sr1 = lb.core.load(path=path1)
y2, sr2 = lb.core.load(path=path2)

length = len(y1)
x1 = np.arange(length)/sr1
x2 = np.arange(2*length, 2*length + len(y2))/sr1

x_train = np.concatenate([x1, x2])[:, np.newaxis]
x_test = np.arange(2 * length + len(y2))[:, np.newaxis]/sr1
y_train = np.concatenate([y1, y2])[:, np.newaxis]
y_test = np.concatenate([y1, np.zeros(length), y2])[:, np.newaxis]

x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

l1 = tf.layers.dense(x, 50, activation=tf.nn.relu)
l1_sin = tf.math.sin(tf.layers.dense(x, 50, activation=None))
l1_cos = tf.math.cos(tf.layers.dense(x, 50, activation=None))
l1_m = tf.math.multiply(
    tf.layers.dense(x, 50, activation=None),
    tf.layers.dense(x, 50, activation=None)
)
l1_final = tf.concat([l1, l1_sin, l1_cos, l1_m], axis=1)
l2 = tf.layers.dense(l1_final, 50, activation=tf.nn.relu)
l2_sin = tf.math.sin(tf.layers.dense(l1_final, 50, activation=None))
l2_cos = tf.math.cos(tf.layers.dense(l1_final, 50, activation=None))
l2_m = tf.math.multiply(
    tf.layers.dense(l1_final, 50, activation=None),
    tf.layers.dense(l1_final, 50, activation=None)
)
l2_final = tf.concat([l2, l2_sin, l2_cos, l2_m], axis=1)
l3 = tf.layers.dense(l2_final, 1, activation=None)

loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=l3))

train_op = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(loss)


tf.set_random_seed(1234)

nbEpochs = 10

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(nbEpochs):
        _, loss_value = sess.run([train_op, loss], feed_dict={x: x_train, y: y_train})
        print("Loss : {0}".format(loss_value))

    predicted = sess.run([l3], feed_dict={x: x_test})

x_test = np.reshape(x_test, -1)
y_test = np.reshape(y_test, -1)
predicted = np.reshape(predicted, -1)

plt.plot(x_test, y_test, label='Real data')
plt.plot(x_test, predicted, label='Predicted Data')
plt.plot([10, 10], [-1, 1], 'k--')
plt.plot([20, 20], [-1, 1], 'k--')
plt.legend()

plt.show()
