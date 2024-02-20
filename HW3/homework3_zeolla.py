import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

SEED = 42
np.random.seed(SEED)

EPOCHS_RANGE = [10, 30]
LEARNING_RATE_RANGE = [0.1, 0.01, 0.001]
BATCH_SIZE_RANGE = [32, 128]
L2_REGULARIZE_RANGE = [0.1, 0.01]

VALIDATION_SIZE = 0.20

TEST_RUN = False
COPY_LOGS = False

if TEST_RUN:
    EPOCHS_RANGE = [50]
    LEARNING_RATE_RANGE = [0.01]
    BATCH_SIZE_RANGE = [32]
    L2_REGULARIZE_RANGE = [0.001]


def problem_3():
    models = []
    n_models = len(EPOCHS_RANGE) * len(LEARNING_RATE_RANGE) * len(BATCH_SIZE_RANGE) * len(L2_REGULARIZE_RANGE)
    for epoch in EPOCHS_RANGE:
        for learning_rate in LEARNING_RATE_RANGE:
            for batch_size in BATCH_SIZE_RANGE:
                for L2_value in L2_REGULARIZE_RANGE:
                    print(f'Running for Model {len(models) + 1}/{n_models}')
                    model_key = f'EPOCH: {epoch}, LR: f{learning_rate}, BATCH: {batch_size}, L2: {L2_value}'
                    validation_metrics, testing_loss, testing_accuracy = linear_regression_MNIST(epoch,
                                                                                                 batch_size, L2_value,
                                                                                                 learning_rate,
                                                                                                 show_epoch_logs=False)
                    model = {
                        'key': model_key,
                        'validation_loss_list': list(map(lambda i: i['loss'], validation_metrics)),
                        'testing_loss': testing_loss,
                        'params': {
                            "EPOCH": epoch,
                            "LR": learning_rate,
                            'BATCH': batch_size,
                            'L2': L2_value
                        },
                        'validation_accuracy_list': list(map(lambda i: i['accuracy'], validation_metrics)),
                        'testing_accuracy': testing_accuracy,
                    }
                    models.append(model)

    filtered_models = list(filter(lambda obj: np.isfinite(obj['testing_loss']), models))
    sorted_models = sorted(filtered_models, key=lambda obj: float(obj['testing_accuracy']), reverse=True)

    print('The Best Model Was: ')
    if len(sorted_models) > 0:
        print(sorted_models[0])
    else:
        print('N/A')

    for i in range(5 if len(sorted_models) > 5 else len(sorted_models)):
        model = sorted_models[i]

        plt.subplot(1, 2, 1)
        plt.plot(model['validation_loss_list'], label=model['key'])
        plt.scatter(model['params']['EPOCH'] - 1, model['testing_loss'],
                    color='red', marker='o')

        plt.text(model['params']['EPOCH'] - 1, model['testing_loss'], f'Testing Loss ', ha='right', va='top')

        plt.xlabel('EPOCH')
        plt.ylabel('CE Loss')
        plt.title('Validation CE vs EPOCH')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(model['validation_accuracy_list'], label=model['key'])
        plt.scatter(model['params']['EPOCH'] - 1, model['testing_accuracy'],
                    color='red', marker='o')

        plt.text(model['params']['EPOCH'] - 1, model['testing_accuracy'], f'Testing Accuracy ', ha='right', va='top')

        plt.xlabel('EPOCH')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy vs EPOCH')

        plt.legend()
        plt.show()


def softmax(y):
    # Apply logsumexp trick
    log_sum = y - np.max(y, axis=1, keepdims=True)
    exp_log_sum = np.exp(log_sum)
    soft_sum = np.sum(exp_log_sum, axis=1, keepdims=True)
    return exp_log_sum / soft_sum


def find_loss_and_gradient(X, y_preds, y_actual, loss_type='MSE', has_gradient=False):
    n = len(y_actual)
    loss = None
    gradient_w = None
    gradient_b = None
    accuracy = None

    if loss_type == 'MSE':
        loss = (1 / (2 * n)) * np.sum((y_preds - y_actual) ** 2)
        if has_gradient:
            gradient_w = (1 / n) * (X.T @ (y_preds - y_actual))  # The @ symbol handles the sum part
            gradient_b = (1 / n) * np.sum(y_preds - y_actual)
    elif loss_type == 'MAE':
        loss = (1 / n) * np.sum(np.abs(y_preds - y_actual))
        if has_gradient:
            gradient_w = (1 / n) * (
                    X.T @ np.where(y_preds != y_actual, ((y_preds - y_actual) / np.abs(y_preds - y_actual)), 0))
            gradient_b = (1 / n) * np.sum(
                (np.where(y_preds != y_actual, ((y_preds - y_actual) / np.abs(y_preds - y_actual)), 0)))
    elif loss_type == 'CE':
        loss = -(1 / n) * np.sum(y_actual * np.log(y_preds))

        class_preds = np.argmax(y_preds, axis=1)
        y_actual_classes = np.argmax(y_actual, axis=1)
        right_preds = np.sum(class_preds == y_actual_classes)
        accuracy = right_preds / len(y_actual)

        # Gradient of W = d(ce)/d(w) => d(ce)/d(y_pred) * d(y_pred)/d(z) * d(z)/d(w)
        # d(ce)/d(y_pred) aka Cross Entropy => d(y * log(y_pred)) => y * 1/y_pred
        # d(y_pred)/d(z) aka SoftMax => y_pred * (1 - y_pred)
        # d(z)/d(w) aka X.T @ w + b => X.T
        # So d(ce)/d(w) = y * 1/y_pred * y_pred * (1 - y_pred) @ X.T
        # Simplify: y * (1 - y_pred) @ X.T
        # Simplify:  (y - y_pred) @ X.T

        # Gradient of b = d(ce)/d(b) => d(ce)/d(y_pred) * d(y_pred)/d(z) * d(z)/d(b)
        # d(ce)/d(y_pred) aka Cross Entropy => d(y * log(y_pred)) => y * 1/y_pred
        # d(y_pred)/d(z) aka SoftMax => y_pred * (1 - y_pred)
        # d(z)/d(b) aka X.T @ w + b => 1
        # So d(ce)/d(b) = y/y_pred * (y_pred - y_pred^2)
        # Simplify: y * (1 - y_pred) -- You can remove the expanded y * y_pred in y - y_pred * y b/c y will  be 1 or 0
        # Simplify: y - y_pred

        if has_gradient:
            gradient_w = -(1 / n) * (X.T @ (y_actual - y_preds))
            gradient_b = -(1 / n) * np.sum(y_actual - y_preds)

    return (loss, accuracy), (gradient_w, gradient_b)


def linear_regression_MNIST(EPOCHS, BATCH_SIZE, L2_REGULARIZE, LEARNING_RATE, show_epoch_logs=False):
    print('\n')
    print(
        f'Running Linear Regression with EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, LEARNING_RATE={LEARNING_RATE} and '
        f'L2_REGULARIZE={L2_REGULARIZE}')

    X_train = np.load('fashion_mnist_train_images.npy')
    y_train = np.load('fashion_mnist_train_labels.npy')

    X_test = np.load('fashion_mnist_test_images.npy')
    y_test = np.load('fashion_mnist_test_labels.npy')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=SEED)

    columns = X_train.shape[1]

    classes = 10
    w = 0.01 * np.random.randn(columns, classes)
    b = 0.01 * np.random.randn(classes)
    validation_per_epoch = []

    y_test = np.eye(classes)[y_test]
    y_val = np.eye(classes)[y_val]
    y_train = np.eye(classes)[y_train]

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    print(f'{X_train.shape} train samples')
    print(f'{X_val.shape} validation samples')
    print(f'{X_test.shape} test samples')
    print('\n')

    print(f'{y_train.shape} train labels')
    print(f'{y_val.shape} validation labels')
    print(f'{y_test.shape} test labels')
    print('\n')

    for EPOCH in range(1, EPOCHS + 1):
        data_indices = np.arange(len(X_train))
        np.random.shuffle(data_indices)
        X_train_shuffled = X_train[data_indices]
        y_train_shuffled = y_train[data_indices]

        for batch_start in range(0, len(X_train_shuffled), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE

            x_batch = X_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]

            batch_preds = x_batch @ w + b
            y_probs = softmax(batch_preds)

            (train_loss, train_acc), (gradient_w, gradient_b) = find_loss_and_gradient(x_batch, y_probs, y_batch,
                                                                                       loss_type='CE',
                                                                                       has_gradient=True)

            if show_epoch_logs:
                print('Train Loss: ', train_loss)
                print('Train Accuracy: ', str(train_acc * 100) + '%')

            w = w - LEARNING_RATE * (gradient_w + L2_REGULARIZE * w)
            b = b - LEARNING_RATE * gradient_b

        validation_preds = X_val @ w + b
        validation_probs = softmax(validation_preds)

        (validation_loss, validation_acc), (_, _) = find_loss_and_gradient(X_val, validation_probs,
                                                                           y_val, loss_type='CE')

        print(f"Epoch {EPOCH}, Validation Loss: {validation_loss} and Validation Accuracy: {validation_acc}")

        validation_per_epoch.append({'loss': validation_loss, 'accuracy': validation_acc})

    testing_preds = X_test @ w + b
    testing_probs = softmax(testing_preds)

    (testing_loss, testing_accuracy), (_, _) = find_loss_and_gradient(X_test, testing_probs, y_test, loss_type='CE')

    print(
        f"Final Validation Loss: {validation_per_epoch[-1]['loss']} and Validation Accuracy: {str(validation_per_epoch[-1]['accuracy'] * 100) + '%'}")
    print(f"Final Testing Loss: {testing_loss} and Testing Accuracy: {str(testing_accuracy * 100) + '%'}")
    print('\n')
    return validation_per_epoch, testing_loss, testing_accuracy


if COPY_LOGS:
    original_stdout = sys.stdout
    logs_path = 'run_logs_linear.txt'
    with open(logs_path, 'w') as log_file:
        sys.stdout = log_file
        print('\nTesting problem_3:')
        problem_3()
        print('\n')
    sys.stdout = original_stdout
else:
    print('\nTesting problem_3:')
    problem_3()
    print('\n')


def AffineTransformation(W, b, x):
    return W @ x + b


def Composition(L, x):
    z = []
    x_input = x
    for w, b in L:
        x_input = AffineTransformation(w, b, x_input)
        z.append(x_input)
    return z


def ComputeGradients(L, x, zs):
    gradients = []
    k = 0
    for (w, b) in L:
        z = x if k == 0 else zs[k - 1]
        db = np.eye(z.shape[0])

        rows, cols = z.T.shape
        dw = []
        for i in range(0, cols * 2):
            row = np.zeros(cols * 2)
            if i <= cols:
                row[i:i + cols] = z.T
                dw.append(row)
        dw = np.array(dw)
        gradients.append((db, dw))
        k += 1
    return gradients


f_w = np.array([1, -2, 1 / 4])
f_b = np.array([0])
f = (f_w, f_b)

g_w = np.array([[1, 2], [0, 1], [-1, 0]])
g_b = np.array([[0], [0], [3]])
g = (g_w, g_b)

x = np.array([[-1], [1]])

L = [g, f]

Z = Composition(L, x)

print('\nTesting problem_4b:')
print('Output: ', AffineTransformation(g_w, g_b, x))
print('\n')

print('\nTesting problem_4d:')
print('Output: ', Z[-1])
print('\n')

gradients = ComputeGradients(L, x, Z)
print('\nTesting problem_4h:')
print('Output: ')
for i, (db, dw) in enumerate(gradients):
    print(f'W_{i}: \n', dw, '\n')
    print(f'b_{i}: \n', db, '\n')
