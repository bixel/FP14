import numpy as np

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, DenseLayer, batch_norm, dropout

from tqdm import tqdm, trange

from sklearn.metrics import roc_auc_score


# Just a small function to randomly chose batches from the dataset
def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def build_net(feature_size=50):
    # first define the input variables as from simple theano objects
    # We use a matrix because we'll use multiple rows of data for training
    input_X = T.matrix(name='Input X')

    # ivector is a simple vector of integer type
    target_Y = T.ivector('Target Y')

    dropout_p = 0.2

    # the input shape is the shape of input_X. None in the first place
    # means the number of rows (batchsize) can be defined later on
    l_in = InputLayer(shape=[None, feature_size],
                      input_var=input_X,
                      name='Input Layer')

    # we can simply add some dense layers
    # (this means each node of the previous layer is connected to every
    # node of this layer. In Matrix-notation this means no matrix element is
    # zero)
    l_1 = DenseLayer(dropout(batch_norm(l_in), dropout_p),
                     num_units=300,
                     name='Dense 1',
                     nonlinearity=lasagne.nonlinearities.tanh)

    l_2 = DenseLayer(dropout(batch_norm(l_1), dropout_p),
                     num_units=300,
                     name='Dense 2',
                     nonlinearity=lasagne.nonlinearities.tanh)

    l_3 = DenseLayer(dropout(batch_norm(l_2), dropout_p),
                     num_units=300,
                     name='Dense 3',
                     nonlinearity=lasagne.nonlinearities.tanh)

    l_4 = DenseLayer(dropout(batch_norm(l_3), dropout_p),
                     num_units=300,
                     name='Dense 4',
                     nonlinearity=lasagne.nonlinearities.tanh)

    l_5 = DenseLayer(dropout(batch_norm(l_4), dropout_p),
                     num_units=300,
                     name='Dense 5',
                     nonlinearity=lasagne.nonlinearities.tanh)

    # finally we need an output layer, we call it nn, since its the essence of
    # the neural net. It is another DenseLayer with only 2 units, one for
    # each class
    # Furthermore we nned the softmax nonlinearity to interpret the output
    # as some kind of probability
    nn = DenseLayer(dropout(batch_norm(l_5), dropout_p),
                    num_units=2,
                    name='Dense Out',
                    nonlinearity=lasagne.nonlinearities.softmax)

    # theano will need to access the weights for optimization
    weights = lasagne.layers.get_all_params(nn, trainable=True)

    # We need to define a loss function
    nn_out = lasagne.layers.get_output(nn)
    loss = lasagne.objectives.categorical_crossentropy(nn_out, target_Y).mean()

    # Furthermore we define the whole training function,
    # which minimize the loss function
    updates = lasagne.updates.adadelta(loss, weights)

    # the theano.function call finally compiles this function
    train_fun = theano.function([input_X, target_Y],
                                [loss, nn_out[:, 1]],
                                updates=updates)

    # Finally we want another function, which wont train, but will predict the
    # data
    det_nn_out = lasagne.layers.get_output(nn, deterministic=True)
    det_loss = lasagne.objectives.categorical_crossentropy(det_nn_out,
                                                           target_Y).mean()
    val_fun = theano.function([input_X, target_Y],
                              [det_loss, det_nn_out[:, 1]])

    return train_fun, val_fun, nn


def iterate_epochs(train_fun, val_fun,
                   train_data, train_labels, val_data, val_labels,
                   epochs=100, batchsize=1000):

    train_auc_curve = []
    val_auc_curve = []

    for epoch in trange(epochs, desc='Epochs'):
        # we'll keep track of some metrics
        train_err = 0
        Ypred_batches = []
        Ytrue_batches = []
        train_batches = 0
        # this is the training batch loop
        for inputs, targets in tqdm(
                iterate_minibatches(
                    train_data.as_matrix(),
                    train_labels.astype('bool').as_matrix(),
                    batchsize,
                    shuffle=True),
                desc='Training Batches',
                total=len(train_data) / batchsize,
                leave=False):
            err, y_pred = train_fun(inputs, targets)
            Ypred_batches.append(y_pred)
            Ytrue_batches.append(targets)

            train_err += err
            train_batches += 1

        Ytrue_train = np.concatenate(Ytrue_batches)
        Ypred_train = np.concatenate(Ypred_batches)
        train_roc_auc = roc_auc_score(Ytrue_train, Ypred_train)

        train_auc_curve.append(train_roc_auc)

        train_err = 0
        Ypred_batches = []
        Ytrue_batches = []
        train_batches = 0

        # this is the validation batch loop
        for inputs, targets in iterate_minibatches(
                val_data.as_matrix(),
                val_labels.astype('bool').as_matrix(),
                batchsize,
                shuffle=False):
            err, y_pred = val_fun(inputs, targets)

            Ypred_batches.append(y_pred)
            Ytrue_batches.append(targets)

            train_err += err
            train_batches += 1

        Ytrue_val = np.concatenate(Ytrue_batches)
        Ypred_val = np.concatenate(Ypred_batches)
        val_roc_auc = roc_auc_score(Ytrue_val, Ypred_val)

        val_auc_curve.append(val_roc_auc)

    return train_auc_curve, val_auc_curve
