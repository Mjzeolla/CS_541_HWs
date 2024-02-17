import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SEED = 42
EPOCHS_RANGE = [10, 15, 25, 100]
LEARNING_RATE_RANGE = [0.1, 0.01, 0.001, 0.0001]
BATCH_SIZE_RANGE = [32, 64, 128, 256]
L2_REGULARIZE_RANGE = [0.1, 0.01, 0.001, 0.0001]
np.random.seed(SEED)
VALIDATION_SIZE = 0.20

TEST_RUN = True
if TEST_RUN:
    EPOCHS_RANGE = [10]
    LEARNING_RATE_RANGE = [0.001]
    BATCH_SIZE_RANGE = [128]
    L2_REGULARIZE_RANGE = [0.01]


def problem_2():
    models = []
    for epoch in EPOCHS_RANGE:
        for learning_rate in LEARNING_RATE_RANGE:
            for batch_size in BATCH_SIZE_RANGE:
                for L2_value in L2_REGULARIZE_RANGE:
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
    sorted_models = sorted(filtered_models, key=lambda obj: float(obj['testing_loss']), reverse=True)

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

        gradient_ce = y_actual * (1 / y_preds)
        gradient_softmax = y_preds * (1 - y_preds)
        gradient_f = X.T

        gradient_w = -(1 / n) * (gradient_f @ (gradient_ce * gradient_softmax))
        gradient_b = -(1 / n) * np.sum(gradient_ce * gradient_softmax)

    return (loss, accuracy), (gradient_w, gradient_b)


def linear_regression_MNIST(EPOCHS, BATCH_SIZE, L2_REGULARIZE, LEARNING_RATE, show_epoch_logs=False):
    X_train = np.load('fashion_mnist_train_images.npy')
    y_train = np.load('fashion_mnist_train_labels.npy')

    X_test = np.load('fashion_mnist_test_images.npy')
    y_test = np.load('fashion_mnist_test_labels.npy')

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=VALIDATION_SIZE, random_state=SEED)

    print(f'{X_train.shape} train samples')
    print(f'{X_val.shape} validation samples')
    print(f'{X_test.shape} test samples')

    columns = X_train.shape[1]

    classes = 10
    w = 0.01 * np.random.randn(columns, classes)
    b = 0.01 * np.random.randn(1)
    validation_per_epoch = []

    y_test = np.eye(classes)[y_test]
    y_val = np.eye(classes)[y_val]
    y_train = np.eye(classes)[y_train]

    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    for EPOCH in range(1, EPOCHS + 1):
        print(f'Running Epoch: {EPOCH}')
        data_indices = np.arange(len(X_train))
        np.random.shuffle(data_indices)
        X_train_shuffled = X_train[data_indices]
        y_train_shuffled = y_train[data_indices]

        for batch_start in range(0, len(X_train_shuffled), BATCH_SIZE):
            batch_end = batch_start + BATCH_SIZE

            x_batch = X_train_shuffled[batch_start:batch_end]
            y_batch = y_train_shuffled[batch_start:batch_end]
            # print(x_batch.shape)
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

        if show_epoch_logs:
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


problem_2()


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

print('\nTesting problem_4d:')
print('Output: ', Z)
print('\n')

gradients = ComputeGradients(L, x, Z)
print('\nTesting problem_4h:')
print('Output: ')
for i, (db, dw) in enumerate(gradients):
    print(f'W_{i}: \n', dw, '\n')
    print(f'b_{i}: \n', db, '\n')
