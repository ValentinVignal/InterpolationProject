import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import librosa as lb
import argparse
import os
import pickle
import json
from math import ceil

def main():
    """
        do everything
    """
    parser = argparse.ArgumentParser(description='Program to interpolate functions between 2 waveforms')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batch to wait before logging training status')
    parser.add_argument('--name', default='',  # Optional, if we want ot name our model
                        help='The name of the model')
    parser.add_argument('--data', type=int, default=1, metavar='N',
                        help='The folder containing the data')
    parser.add_argument('--transition-size', type=int, default=0, metavar='N',
                        help='the size of the transition')
    parser.add_argument('--model', type=str, default='1',
                        help='The model of the Neural Network used for the interpolation')
    parser.add_argument('--batch', type=int, default=2048,
                        help='The size of the batchs')

    args = parser.parse_args()

    path_data_folder = '../Data/data_' + str(args.data)

    # The importation of the data
    path1 = os.path.join(path_data_folder, 'part_1.wav')
    path2 = os.path.join(path_data_folder, 'part_2.wav')
    y1, sr1 = lb.core.load(path=path1, mono=True)
    y2, sr2 = lb.core.load(path=path2, mono=True)

    len1 = len(y1)
    lenTrans = len1 if args.transition_size == 0 else args.transition_size
    len2 = len(y2)

    # Construction of the inputs
    x1 = np.arange(len1) / sr1
    x2 = np.arange(len1 + lenTrans, len1 + lenTrans + len2) / sr1

    x_train = np.concatenate([x1, x2])[:, np.newaxis]
    x_test = np.arange(len1 + lenTrans + len2)[:, np.newaxis] / sr1
    y_train = np.concatenate([y1, y2])[:, np.newaxis]
    y_test = np.concatenate([y1, np.zeros(lenTrans), y2])[:, np.newaxis]

    ##### Neural Network part ####

    # Placeholders
    x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    # Loading the model from the JSON file
    json_path = os.path.join('NN', 'hyp_param', args.model + '.JSON')
    with open(json_path, 'r') as f:
        model = json.load(f)

    # Creating the layers
    layers = [x]
    for i in range(model['nb_layers']):
        layers.append(
            tf.concat([
                tf.layers.dense(layers[-1], model['layers_size']['id'][i], activation=tf.nn.tanh),
                tf.math.sin(tf.layers.dense(layers[-1], model['layers_size']['sin'][i], activation=None)),
                tf.math.cos(tf.layers.dense(layers[-1], model['layers_size']['cos'][i], activation=None)),
                tf.math.multiply(
                    tf.layers.dense(layers[-1], model['layers_size']['mul'][i], activation=tf.nn.tanh),
                    tf.layers.dense(layers[-1], model['layers_size']['mul'][i], activation=tf.nn.tanh)
                )
            ], axis=1)
        )

    # Final Layer
    final_layer = tf.layers.dense(layers[-1], 1, activation=tf.nn.tanh)

    # Loss function and optimizer
    loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y, predictions=final_layer))
    train_op = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
    tf.set_random_seed(args.seed)

    loss_tab = []
    nb_points = x_train.shape[0]

    # Train
    print('Start training')
    use_gpu = (not args.no_cuda) and tf.test.is_gpu_available()
    print('use gpu bool : {0}'.format(use_gpu))
    with tf.Session(config=tf.ConfigProto(log_device_placement=use_gpu)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.epochs):
            j = 0
            for b in range(0, nb_points, args.batch):
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={x: x_train[b: b + args.batch], y: y_train[b:b + args.batch]})
                loss_tab.append(loss_value)
                if j % args.log_interval == 0:
                    print("Epoch {0} [{2}/{3}] -> Loss : {1}".format(i + 1, loss_value, b + args.batch, nb_points))
                j += 1

        predicted = sess.run([final_layer], feed_dict={x: x_test})
    print('Training Done')

    x_test = np.reshape(x_test, -1)
    y_test = np.reshape(y_test, -1)
    predicted = np.reshape(predicted, -1)

    ##### Save part #####

    # Find name and create folder
    i = 0
    nameStr = args.name if args.name == '' else '_' + args.name
    save_name = 'data_{0}_epoch({1}){2}_({3})'.format(args.data, args.epochs, nameStr, i)
    while os.path.isdir(os.path.join(path_data_folder, save_name)):
        i += 1
        save_name = 'data_{0}_epoch({1}){2}_({3})'.format(args.data, args.epochs, nameStr, i)
    path_save_folder = os.path.join(path_data_folder, save_name)
    os.mkdir(path_save_folder)

    # Save all the informations
    with open(os.path.join(path_save_folder, save_name + '.p'), 'wb') as dump_file:
        pickle.dump({
            'loss': loss_tab,
            'name': args.name,
            'epochs': args.epochs,
            'data': args.data,
            'transition_size': len1 if args.transition_size == 0 else args.transition_size,
            'train_data': {
                'x_train': x_train,
                'y_train': y_train},
            'test_data': {
                'x_test': x_test,
                'y_test': y_test},
            'data_1': {
                'len': len1,
                'sr': sr1,
                'y': y1},
            'data_2': {
                'len': len2,
                'sr': sr2,
                'y': y2},
            'predicted': predicted
        }, dump_file)

    # Plot of prediction
    plt.figure()
    plt.plot(x_test, y_test, color='steelblue', label='Real data')
    plt.plot(x_test, predicted, color='darkorange', label='Predicted Data')
    plt.plot([len1/sr1, len1/sr1], [-1, 1], color='dimgray', linestyle='--')
    plt.plot([(len1+lenTrans)/sr1, (len1+lenTrans)/sr1], [-1, 1], color='dimgray', linestyle='--')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Interpolated function')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path_save_folder, 'Prediction_' + save_name + '.png'))

    # Plot of prediction
    nb_batchs = ceil(nb_points/args.batch)

    plt.figure()
    plt.plot(np.arange(args.epochs * nb_batchs)/nb_batchs, loss_tab, color='crimson', label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Evolution of the value of the Loss function through the Epochs')
    plt.grid()
    plt.legend()
    plt.savefig(os.path.join(path_save_folder, 'Loss_' + save_name + '.png'))

    print('Datas saved in : {0}'.format(path_save_folder))

if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
