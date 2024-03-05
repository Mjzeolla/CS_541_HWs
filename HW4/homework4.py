import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras import layers
import matplotlib.pyplot as plt
from copy import deepcopy
import sys

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
COPY_LOGS_PROBLEM_3 = False
COPY_LOGS_PROBLEM_2 = False

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
            gradient_b = -(1 / n) * np.sum(previous_error)  # TODO: add axis=0 for check_grad
        else:
            gradient_r = np.where(z > 0, 1, 0)  # Apply ReLu gradient on inputs to ReLu layer activation
            error = previous_error @ Ws[i + 1].T  # Find error in reference to the weights and previous error
            error = error * gradient_r  # Apply ReLu error to the current layer

            gradient_w = -(1 / n) * h.T @ error  # Find gradient based on ReLu error and inputs
            gradient_b = -(1 / n) * np.sum(error)  # TODO: add axis=0 for check_grad

            previous_error = error  # Update error for next layer to be the current layer error

        dJdWs.append(gradient_w.T)
        dJdbs.append(gradient_b)

        # print(f'The i is: {i}\n')
        # print(f'W: {Ws[i].shape}')
        # print(f'dW: {gradient_w.T.shape}')
        #
        # print('\n')
        # print(f'b: {bs[i].shape}')
        # print(f'db: {gradient_b.shape}')
        Ws[i] = Ws[i] - LEARNING_RATE * (gradient_w.T + L2_REGULARIZE * Ws[i])
        bs[i] = bs[i] - LEARNING_RATE * gradient_b

    weightsAndBiases[:] = np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

    # Concatenate gradients
    return np.hstack([dJdW.flatten() for dJdW in dJdWs] + [dJdb.flatten() for dJdb in dJdbs])


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
        print(f'Running EPOCH {epoch + 1}/{NUM_EPOCHS}...')

        # SGD Randomization for indices
        X_train_shuffled, y_train_shuffled = SGD_date_shuffle(trainX, trainY)

        for batch_start in range(0, len(X_train_shuffled), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE

            x_batch = X_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]

            _ = back_prop(x_batch, y_batch, weightsAndBiases)
            trajectory.append(deepcopy(weightsAndBiases))

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
    pca = PCA(n_components=2)
    pca_fit = pca.fit_transform(trajectory)
    fig = plt.figure()

    dimensions_width = 20

    # TODO: Work on the linspace
    min_x, min_y, max_x, max_y = min(pca_fit[:, 0]), min(pca_fit[:, 1]), max(pca_fit[:, 0]), max(pca_fit[:, 1])
    axis1 = np.linspace(min_x, max_x, dimensions_width)
    axis2 = np.linspace(min_y, max_y, dimensions_width)

    Xaxis, Yaxis = np.meshgrid(axis1, axis2)
    Zaxis = np.zeros((len(axis1), len(axis2)))

    for i in range(len(axis1)):
        for j in range(len(axis2)):
            pca_inverse = pca.inverse_transform([Xaxis[i, j], Yaxis[i, j]])
            loss, _, _, _, _ = forward_prop(trainX, trainY, pca_inverse)
            Zaxis[i, j] = loss

    ax = fig.add_subplot(projection='3d')
    ax.plot_surface(Xaxis, Yaxis, Zaxis, alpha=0.6)
    ax.set_zlabel('CE Loss')

    amt_points = 100  # Define Number of Loss Plots to plot

    # Choose a random set of weight params from trajectory
    chosen_params = np.random.choice(len(trajectory), size=amt_points, replace=False)
    chosen_params = [trajectory[i] for i in chosen_params]

    # PCA Transform the selected weight/bias parameters
    pca_params_fit = pca.transform(chosen_params)  # Apply the already fitted model from the previous dataset
    x_points, y_points = pca_params_fit[:, 0], pca_params_fit[:, 1]
    Xaxis, Yaxis = x_points, y_points
    Zaxis = np.zeros(len(chosen_params))

    # Find Loss (Z-Axis) for each weight combination
    for i, param in enumerate(chosen_params):
        loss, _, _, _, _ = forward_prop(trainX, trainY, param)
        Zaxis[i] = loss

    ax.scatter(Xaxis, Yaxis, Zaxis, color='r')
    plt.show()


def plot_training_history(history, epochs, plot_metric="accuracy", plot_metric_label="Accuracy", has_validation=True,
                          testing_loss=None, testing_accuracy=None, epoch_reduce=None, suptitle=None):
    acc = history.history[plot_metric][-epoch_reduce:] if epoch_reduce else history.history[plot_metric]
    loss = history.history['loss'][-epoch_reduce:]if epoch_reduce else history.history['loss']

    if has_validation:
        val_acc = history.history['val_' + plot_metric][-epoch_reduce:] if epoch_reduce else history.history['val_' + plot_metric]
        val_loss = history.history['val_loss'][-epoch_reduce:] if epoch_reduce else history.history['val_loss']

    if epoch_reduce:
        epochs_range = range(epochs - epoch_reduce, epochs)
    else:
        epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training ' + plot_metric_label)
    if has_validation:
        plt.plot(epochs_range, val_acc, label='Validation ' + plot_metric_label)
    plt.legend(loc='lower right')
    plt.title('Training and Validation ' + plot_metric_label)
    plt.xlabel('EPOCHS')

    if testing_accuracy:
        plt.scatter(epochs_range[-1], testing_accuracy, color='red', marker='o')
        plt.text(epochs_range[-1], testing_accuracy, f'Testing Accuracy {testing_accuracy * 100:.0f}%', ha='right', va='top')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    if has_validation:
        plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.xlabel('EPOCHS')
    plt.title('Training and Validation Loss')

    if testing_loss:
        plt.scatter(epochs_range[-1], testing_loss, color='red', marker='o')
        plt.text(epochs_range[-1], testing_loss, f'Testing Loss {testing_loss:.2f}', ha='right', va='top')

    if suptitle:
        plt.suptitle(suptitle)

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
    NUM_EPOCHS = 100
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = setup_MNIST(has_validation=True)

    if MODEL_ARCHITECTURE[0]['input_shape'] is None:
        ValueError('No input_shape passed to first layer')

    model = tf_build_model(MODEL_ARCHITECTURE, L2_REGULARIZE)

    optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    training_history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE,
                                 epochs=NUM_EPOCHS)

    (loss, accuracy) = model.evaluate(X_test, y_test)
    print("Testing loss:", loss)
    print(f"Testing accuracy: {accuracy * 100}%")

    plot_training_history(training_history, NUM_EPOCHS)


def findBestHyperparameters():
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = setup_MNIST(has_validation=True)

    if TEST_RUN:
        HIDDEN_LAYERS = [5]
        NEURON_SIZES = [50]
        LEARNING_RATES = [0.001]
        BATCH_SIZES = [64]
        EPOCHS_RANGE = [200]
        L2_REGULARIZE_RANGE = [0.001]
    else:
        HIDDEN_LAYERS = [3, 4, 5]
        NEURON_SIZES = [40, 50]
        LEARNING_RATES = [0.001]
        BATCH_SIZES = [64]
        EPOCHS_RANGE = [220]
        L2_REGULARIZE_RANGE = [0.001]

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
                                'batch_size': batch_size,
                                'architecture': architecture,
                                'l2_strength': l2_strength,
                                'learning_rate': learning_rate,
                                'testing_results': {
                                    'loss': loss,
                                    'accuracy': accuracy
                                },
                                'validation_results': {
                                    'loss': training_history.history['val_loss'][-1],
                                    'accuracy': training_history.history['val_accuracy'][-1]
                                }
                            })

                            # plot_training_history(training_history, epochs)
        return models


def problem_3c():
    print('\nTesting problem_3b:')
    models = findBestHyperparameters()
    sorted_models_by_validation = sorted(models, key=lambda obj: float(obj['validation_results']['loss']),
                                         reverse=True)
    print('The Best Model Was: ')
    if len(sorted_models_by_validation) > 0:
        best_model = sorted_models_by_validation[0]
        print(best_model)
    else:
        print('N/A')

    print('\nTesting problem_3c:')
    print('The model being used is:')
    if len(sorted_models_by_validation) > 0:
        best_model = sorted_models_by_validation[0]
        print(best_model)

        print('\nRe-training the best model!\n')
        model = tf_build_model(best_model['architecture'], best_model['l2_strength'])

        optimizer = tf.keras.optimizers.SGD(learning_rate=best_model['learning_rate'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        model.summary()

        print(
            f'\nTraining NN with EPOCHS={best_model["epochs"]}, BATCH_SIZE={best_model["batch_size"]}, LEARNING_RATE={best_model["learning_rate"]} and '
            f'L2_REGULARIZE={best_model["l2_strength"]}, ARCHITECTURE={best_model["architecture"]}')

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = setup_MNIST(has_validation=True)

        training_history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                                     batch_size=best_model["batch_size"], epochs=best_model["epochs"])

        (loss, accuracy) = model.evaluate(X_test, y_test)
        print("Testing loss:", loss)
        print(f"Testing accuracy: {accuracy * 100}%")

        epochs = best_model['epochs']
        plot_training_history(training_history, epochs, testing_loss=loss, testing_accuracy=accuracy,
                              suptitle='All Training Epochs')

        plot_training_history(training_history, epochs, testing_loss=loss, testing_accuracy=accuracy,
                              epoch_reduce=20, suptitle='Last 20 Training Epochs')
    else:
        print('N/A')


if __name__ == "__main__":
    (trainX, trainY), (_, _), (testX, testY) = setup_MNIST()

    # Initialize weights and biases randomly
    weightsAndBiases = initWeightsAndBiases()

    # Perform gradient check on 5 training examples
    ##TODO: DO this
    print('The check_grad value is:')
    # print(scipy.optimize.check_grad(lambda wab: forward_prop(np.atleast_2d(trainX[0:5, :]), np.atleast_2d(trainY[0:5, :]), wab)[0], \
    #                                 lambda wab: back_prop(np.atleast_2d(trainX[0:5, :]), np.atleast_2d(trainY[0:5, :]), wab)[0], \
    #                                 weightsAndBiases))

    weightsAndBiases, trajectory = train(trainX, trainY, weightsAndBiases, testX, testY)

    # Plot the SGD trajectory
    # TODO: DO PART B of problem 4 (it should be done now)
    plotSGDPath(trainX, trainY, trajectory)

    if COPY_LOGS_PROBLEM_3:
        original_stdout = sys.stdout
        logs_path = 'run_problem_3_logs.txt'
        with open(logs_path, 'w') as log_file:
            sys.stdout = log_file
            problem_3a()
            problem_3c()
            print('\n')
        sys.stdout = original_stdout
    else:
        problem_3a()
        problem_3c()
        print('\n')