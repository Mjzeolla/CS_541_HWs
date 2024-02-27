import copy

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers

# For this assignment, assume that every hidden layer has the same number of neurons.
NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = 10
NUM_OUTPUT = 10
VALIDATION_SIZE = 0.2
SEED = 0
LEARNING_RATE = 0.01
BATCH_SIZE = 64
L2_REGULARIZE = 0.01
CLASSES = 10

np.random.seed(SEED)

TEST_RUN = False
MODEL_ARCHITECTURE = [
    {
        'name': 'input_layer',
        'size': 10,
        'activation_function': 'relu',
        'input_shape': (784,)
    },
    {
        'name': 'hidden_1',
        'size': 10,
        'activation_function': 'relu'
    },
    {
        'name': 'hidden_3',
        'size': 10,
        'activation_function': 'relu'
    },
    {
        'name': 'output_layer',
        'size': CLASSES,
        'activation_function': 'softmax'
    },
]

# Unpack a list of weights and biases into their individual np.arrays.
def unpack(weightsAndBiases):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT * NUM_HIDDEN
    W = weightsAndBiases[start:end]
    Ws.append(W)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN * NUM_HIDDEN
        W = weightsAndBiases[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN * NUM_OUTPUT
    W = weightsAndBiases[start:end]
    Ws.append(W)

    Ws[0] = Ws[0].reshape(NUM_HIDDEN, NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        # Convert from vectors into matrices
        Ws[i] = Ws[i].reshape(NUM_HIDDEN, NUM_HIDDEN)
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN)

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN
    b = weightsAndBiases[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN
        b = weightsAndBiases[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weightsAndBiases[start:end]
    bs.append(b)

    return Ws, bs


def softmax(y):
    # Apply logsumexp trick
    log_sum = y - np.max(y, axis=1, keepdims=True)
    exp_log_sum = np.exp(log_sum)
    soft_sum = np.sum(exp_log_sum, axis=1, keepdims=True)
    return exp_log_sum / soft_sum


def relu(y):
    return np.maximum(0.0, y)


def relu_gradient(z):
    return np.where(z > 0, 1, 0)


def find_loss(y_preds, y_actual, loss_type='MSE'):
    n = len(y_actual)
    loss = None
    accuracy = None

    if loss_type == 'MSE':
        loss = (1 / (2 * n)) * np.sum((y_preds - y_actual) ** 2)
    elif loss_type == 'MAE':
        loss = (1 / n) * np.sum(np.abs(y_preds - y_actual))
    elif loss_type == 'CE':
        loss = -(1 / n) * np.sum(y_actual * np.log(y_preds))

        class_preds = np.argmax(y_preds, axis=1)
        y_actual_classes = np.argmax(y_actual, axis=1)
        right_preds = np.sum(class_preds == y_actual_classes)
        accuracy = right_preds / len(y_actual)

    return loss, accuracy


def forward_prop(x, y, weightsAndBiases):
    Ws, bs = unpack(weightsAndBiases)
    hs = [x]
    zs = []

    for layer, (w, b) in enumerate(zip(Ws, bs)):
        z = hs[layer] @ w.T + b
        zs.append(z)

        if len(Ws) - 1 == layer:
            h = softmax(z)
        else:
            h = relu(z)

        hs.append(h)

    yhat = hs[len(Ws)]
    loss, acc = find_loss(yhat, y, 'CE')

    # Return loss, pre-activations, post-activations, and predictions
    return loss, zs, hs, yhat, acc


def back_prop(x, y, weightsAndBiases):
    loss, zs, hs, yhat, acc = forward_prop(x, y, weightsAndBiases)
    Ws, bs = unpack(weightsAndBiases)
    dJdWs = []  # Gradients w.r.t. weights
    dJdbs = []  # Gradients w.r.t. biases

    n = len(y)
    previous_error = y - yhat

    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        h = hs[i]  # Input to the current layer (aka output from previous)
        z = zs[i]  # Linear Affine Transformation based on input H

        if i == NUM_HIDDEN_LAYERS:
            gradient_w = -(1 / n) * (h.T @ previous_error)
            gradient_b = -(1 / n) * np.sum(previous_error)
        else:

            gradient_a = relu_gradient(z)  # Apply ReLu gradient on inputs to ReLu layer activation
            error = previous_error @ Ws[i + 1].T  # Find error in reference to the weights and previous error
            error = error * gradient_a  # Apply ReLu error to the current layer

            gradient_w = -(1 / n) * h.T @ error  # Find gradient based on ReLu error and inputs
            gradient_b = -(1 / n) * np.sum(error)

            previous_error = error  # Update error for next layer to be the current layer error

        dJdWs.append(gradient_w)
        dJdbs.append(gradient_b)

        Ws[i] = Ws[i] - LEARNING_RATE * (gradient_w.T + L2_REGULARIZE * Ws[i])
        bs[i] = bs[i] - LEARNING_RATE * gradient_b

    # Concatenate gradients
    return np.hstack([dJdW.flatten() for dJdW in dJdWs] + [dJdb.flatten() for dJdb in dJdbs]), np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])


def SGD_date_shuffle(X_train, y_train):
    data_indices = np.arange(len(X_train))
    np.random.shuffle(data_indices)
    return X_train[data_indices], y_train[data_indices]


def train(trainX, trainY, weightsAndBiases, testX, testY):
    NUM_EPOCHS = 100
    trajectory = []

    print(
        f'\nTraining Model with EPOCHS={NUM_EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE} and '
        f'L2_REGULARIZE={L2_REGULARIZE}')

    for epoch in range(NUM_EPOCHS):
        # TODO: save the current set of weights and biases into trajectory; this is
        # useful for visualizing the SGD trajectory.

        print(f'Running EPOCH {epoch+1}/{NUM_EPOCHS}...')

        # SGD Randomization for indices
        X_train_shuffled, y_train_shuffled = SGD_date_shuffle(trainX, trainY)

        for batch_start in range(0, len(X_train_shuffled), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE

            x_batch = X_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]

            gradients, weights = back_prop(x_batch, y_batch, weightsAndBiases)
            weightsAndBiases = weights
            trajectory.append(gradients)

    testing_loss, _, _, _, testing_accuracy = forward_prop(testX, testY, weightsAndBiases)

    print(f"\nTesting Loss: {testing_loss} and Testing Accuracy: {testing_accuracy}")

    return weightsAndBiases, trajectory


# Performs a standard form of random initialization of weights and biases
def initWeightsAndBiases():
    Ws = []
    bs = []

    np.random.seed(SEED)
    W = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN)
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN)
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN)) / NUM_HIDDEN ** 0.5) - 1. / NUM_HIDDEN ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])


def setup_MNIST(classes=10, has_validation=False):
    print('\nImporting MNIST Dataset...')
    X_train = np.load('fashion_mnist_train_images.npy')
    y_train = np.load('fashion_mnist_train_labels.npy')

    X_test = np.load('fashion_mnist_test_images.npy')
    y_test = np.load('fashion_mnist_test_labels.npy')

    if has_validation:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE,
                                                          random_state=SEED)
    else:
        X_val = None
        y_val = None

    # One-Hot-Encode Labels
    y_test = np.eye(classes)[y_test]

    if has_validation:
        y_val = np.eye(classes)[y_val]

    y_train = np.eye(classes)[y_train]

    # Normalize Inputs by 255.0
    X_train = X_train / 255.0 - 0.5

    if has_validation:
        X_val = X_val / 255.0 - 0.5

    X_test = X_test / 255.0 - 0.5

    print(f'{X_train.shape} train samples')

    if has_validation:
        print(f'{X_val.shape} validation samples')

    print(f'{X_test.shape} test samples')
    print('\n')

    print(f'{y_train.shape} train labels')

    if has_validation:
        print(f'{y_val.shape} validation labels')
    print(f'{y_test.shape} test labels')
    print('\n')

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def plotSGDPath(trainX, trainY, trajectory):
    # TODO: change this toy plot to show a 2-d projection of the weight space
    # along with the associated loss (cross-entropy), plus a superimposed 
    # trajectory across the landscape that was traversed using SGD. Use
    # sklearn.decomposition.PCA's fit_transform and inverse_transform methods.

    def toyFunction(x1, x2):
        return np.sin((2 * x1 ** 2 - x2) / 10.)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Compute the CE loss on a grid of points (corresonding to different w).
    axis1 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    axis2 = np.arange(-np.pi, +np.pi, 0.05)  # Just an example
    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))
    for i in range(len(axis1)):
        for j in range(len(axis2)):
            Zaxis[i, j] = toyFunction(Xaxis[i, j], Yaxis[i, j])
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)  # Keep alpha < 1 so we can see the scatter plot too.

    # Now superimpose a scatter plot showing the weights during SGD.
    Xaxis = 2 * np.pi * np.random.random(8) - np.pi  # Just an example
    Yaxis = 2 * np.pi * np.random.random(8) - np.pi  # Just an example
    Zaxis = toyFunction(Xaxis, Yaxis)
    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')

    plt.show()

def plot_training_history(history, epochs, plot_metric="accuracy", plot_metric_label="Accuracy", has_validation=True):
    acc = history.history[plot_metric]
    loss = history.history['loss']

    if has_validation:
        val_acc = history.history['val_' + plot_metric]
        val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training ' + plot_metric_label)
    if has_validation:
        plt.plot(epochs_range, val_acc, label='Validation ' + plot_metric_label)
    plt.legend(loc='lower right')
    plt.title('Training and Validation ' + plot_metric_label)
    plt.xlabel('EPOCHS')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if has_validation:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('EPOCHS')
    plt.title('Training and Validation Loss')
    plt.show()


def tf_build_model(architecture, l2_rate):
    if architecture[0]['input_shape'] is None:
        ValueError('No input_shape passed to first layer')

    model = keras.models.Sequential()

    for index, layer in enumerate(architecture):
        if layer['size'] is None:
            ValueError(f'No neuron size passed to {index} layer')
        elif layer['activation_function'] is None:
            ValueError(f'No activation_function passed to {index} layer')

        if index == 0:
            model.add(layers.Dense(layer['size'], layer['activation_function'], input_shape=layer['input_shape'],
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_rate), name=layer['name'])
                      )
        else:
            model.add(layers.Dense(layer['size'], layer['activation_function'], name=layer['name'],
                                   kernel_regularizer=tf.keras.regularizers.l2(l2_rate))
                      )
    return model


def problem_3a():
    print('\nTesting problem_3a:')
    NUM_EPOCHS = 50
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = setup_MNIST()

    if MODEL_ARCHITECTURE[0]['input_shape'] is None:
        ValueError('No input_shape passed to first layer')

    model = tf_build_model(MODEL_ARCHITECTURE, L2_REGULARIZE)

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    training_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)

    (loss, accuracy) = model.evaluate(X_test, y_test)
    print("Testing loss:", loss)
    print(f"Testing accuracy: {accuracy * 100}%")

    plot_training_history(training_history, NUM_EPOCHS)


def findBestHyperparameters():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = setup_MNIST()

    if TEST_RUN:
        HIDDEN_LAYERS = [10]
        NEURON_SIZES = [300]
        LEARNING_RATES = [0.01]
        BATCH_SIZES = [32]
        EPOCHS_RANGE = [100]
        L2_REGULARIZE_RANGE = [0.01]
    else:
        HIDDEN_LAYERS = [3, 4, 5]
        NEURON_SIZES = [40, 50]
        LEARNING_RATES = [0.01]
        BATCH_SIZES = [64]
        EPOCHS_RANGE = [50]
        L2_REGULARIZE_RANGE = [0.01]

    n_models = len(L2_REGULARIZE_RANGE) * len(EPOCHS_RANGE) * len(HIDDEN_LAYERS) * len(NEURON_SIZES) * len(
        LEARNING_RATES) * len(BATCH_SIZES)

    models = []
    for epochs in EPOCHS_RANGE:
        for l2_strength in L2_REGULARIZE_RANGE:
            for batch_size in BATCH_SIZES:
                for learning_rate in LEARNING_RATES:
                    for hidden_layer_size in HIDDEN_LAYERS:
                        for neuron_size in NEURON_SIZES:
                            print(f'\nRunning for Model {len(models) + 1}/{n_models} \n')
                            model_key = f'EPOCH: {epochs}, LAYER_AMT: {hidden_layer_size}, NEURONS: {neuron_size}, ' \
                                        f'LR: {learning_rate}, BATCH: {batch_size}, L2: {l2_strength}'

                            architecture = [{
                                'name': 'input_layer',
                                'size': neuron_size,
                                'activation_function': 'relu',
                                'input_shape': (784,)
                            }]

                            for index in range(1, hidden_layer_size + 1):
                                layer = {
                                    'name': f'hidden{index}',
                                    'size': neuron_size,
                                    'activation_function': 'relu'
                                }
                                architecture.append(layer)

                            architecture.append({
                                'name': 'output_layer',
                                'size': CLASSES,
                                'activation_function': 'softmax',
                            })

                            model = tf_build_model(architecture, l2_strength)

                            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
                            model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

                            model.summary()

                            print(
                                f'\nTraining NN with EPOCHS={epochs}, BATCH_SIZE={batch_size}, LEARNING_RATE={learning_rate} and '
                                f'L2_REGULARIZE={l2_strength}, ARCHITECTURE={architecture}')

                            training_history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                                         batch_size=batch_size, epochs=epochs)

                            (loss, accuracy) = model.evaluate(X_test, y_test)
                            print("Testing loss:", loss)
                            print(f"Testing accuracy: {accuracy * 100}%")

                            models.append({
                                'key': model_key,
                                'model': model,
                                'history': training_history,
                                'epochs': epochs,
                                'l2_strength': l2_strength,
                                'testing_results': {
                                    'loss': loss,
                                    'accuracy': accuracy
                                },
                                'validation_results': {
                                    'loss': training_history.history['val_loss'][-1],
                                    'accuracy': training_history.history['val_accuracy'][-1]
                                }
                            })

                            plot_training_history(training_history, epochs)
        return models


def problem_3b():
    print('\nTesting problem_3b:')
    models = findBestHyperparameters()
    sorted_models_by_validation = sorted(models, key=lambda obj: float(obj['validation_results']['accuracy']),
                                         reverse=True)

    print('\nTesting problem_3c:')
    print('The Best Model Was: ')
    if len(sorted_models_by_validation) > 0:
        best_model = sorted_models_by_validation[0]
        print(best_model)
        best_training_history, epochs = best_model['history'], best_model['epochs']
        plot_training_history(best_training_history, epochs)
    else:
        print('N/A')

if __name__ == "__main__":
    (trainX, trainY), (_, _), (testX, testY) = setup_MNIST()

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

    # Perform gradient check on 5 training examples
    ##TODO: DO this
    # print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab)[0], \
    #                                 lambda wab: back_prop(np.atleast_2d(trainX[:,0:5]), np.atleast_2d(trainY[:,0:5]), wab), \
    #                                 weightsAndBiases))

    weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)

    # Plot the SGD trajectory
    ##TODO: DO this
    plotSGDPath(trainX, trainY, trajectory)
