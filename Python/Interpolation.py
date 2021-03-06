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
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1234)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batch to wait before logging training status')
    parser.add_argument('--name', default='',  # Optional, if we want ot name our model
                        help='The name of the model')
    parser.add_argument('--data', type=str, default='1', metavar='N',
                        help='The folder containing the data')
    parser.add_argument('--transition-size', type=int, default=0, metavar='N',
                        help='the size of the transition')
    parser.add_argument('--model', type=str, default='1',
                        help='The model of the Neural Network used for the interpolation')
    parser.add_argument('--batch', type=int, default=1,
                        help='The number of the batchs')
    parser.add_argument('--small-cpu', action='store_true', default=False,
                        help='To work on small CPU')
    parser.add_argument('--gpu', type=str, default='0',
                        help='What GPU to use')

    args = parser.parse_args()
    if args.small_cpu:
        args.batch = 1000
        args.no_cuda = True

    path_data_folder = '../Data/data_' + args.data

    tf.device('/device:GPU:' + args.gpu) if not args.no_cuda else None

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

    # FC Layers
    for i in range(model['nb_fc_layers']):
        layers.append(tf.layers.dense(layers[-1], model['fc_layers_size'][i], activation=tf.nn.tanh))

    # Final Layer
    final_layer = tf.layers.dense(layers[-1], 1, activation=tf.nn.tanh)

    # Loss function and optimizer
    loss = tf.reduce_sum(
        tf.losses.mean_squared_error(labels=y, predictions=final_layer, reduction=tf.losses.Reduction.SUM))
    train_op = tf.train.AdamOptimizer(learning_rate=args.lr).minimize(loss)
    tf.set_random_seed(args.seed)

    loss_epoch = []
    nb_points = x_train.shape[0]
    batch_size = int(ceil(nb_points/args.batch))

    # Train
    print('Start training')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(args.epochs):
            loss_epoch.append(0)
            for b in range(args.batch):
                _, loss_value = sess.run([train_op, loss],
                                         feed_dict={x: x_train[b * batch_size: min((b + 1) * batch_size, nb_points)],
                                                    y: y_train[b * batch_size: min((b + 1) * batch_size, nb_points)]})
                loss_epoch[-1] += loss_value
            if i % args.log_interval == 0:
                print("Epoch {0} -> Loss : {1}".format(i + 1, loss_epoch[-1]))

        predicted = sess.run([final_layer], feed_dict={x: x_test})
    print('Training Done')

    x_test = np.reshape(x_test, -1)
    y_test = np.reshape(y_test, -1)
    predicted = np.reshape(predicted, -1)

    ##### Save part #####
    path_save_folder = '../SavedFolder'
    path_save_folder_data = os.path.join(path_save_folder, 'data_' + args.data)
    os.mkdir(path_save_folder) if not os.path.isdir(path_save_folder) else None
    os.mkdir(path_save_folder_data) if not os.path.isdir(path_save_folder_data) else None


    # Find name and create folder
    i = 0
    nameStr = args.name if args.name == '' else '_' + args.name
    save_name = 'data_{0}_epoch({1}){2}_({3})'.format(args.data, args.epochs, nameStr, i)
    while os.path.isdir(os.path.join(path_save_folder_data, save_name)):
        i += 1
        save_name = 'data_{0}_epoch({1}){2}_({3})'.format(args.data, args.epochs, nameStr, i)
    path_to_save_folder = os.path.join(path_save_folder_data, save_name)
    os.mkdir(path_to_save_folder)

    # Save all the informations
    with open(os.path.join(path_to_save_folder, save_name + '.p'), 'wb') as dump_file:
        pickle.dump({
            'loss': loss_epoch,
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
    plt.savefig(os.path.join(path_to_save_folder, 'Prediction_' + save_name + '.png'))

    # Plot of prediction

    color_epoch = 'tab:blue'
    plt.plot()

    plt.plot(np.arange(1, args.epochs + 1), loss_epoch, color=color_epoch, label='Loss through the epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.title('Evolution of the value of the Loss function\nthrough the Epochs and Batchs')
    plt.grid()
    plt.savefig(os.path.join(path_to_save_folder, 'Loss_' + save_name + '.png'))


    # Save of the .wav file
    lb.output.write_wav(os.path.join(path_to_save_folder, save_name + '.wav'), np.reshape(y_train, -1), sr1)

    print('Datas saved in : {0}'.format(path_to_save_folder))


if __name__ == '__main__':
    # create a separate main function because original main function is too mainstream
    main()
