'''
Chaoyu Li
Date: 3/30/2022
'''
import numpy as np
import cv2

train_data = 'downgesture_train.list'
test_data = 'downgesture_test.list'
num_neurons = 100
num_epochs = 1000
lr = 0.1
np.random.seed(8)

def label(filename):
    label1_list = []
    label0_list = []
    with open(filename) as f:
        for line in f.readlines():
            if 'down' not in line:
                label0_list.append(line.strip())
            else:
                label1_list.append(line.strip())
    return label1_list, label0_list

def readImage(files, label):
    output = []
    for file in files:
        img = cv2.imread(file, 0)
        # reshape the image to 1-D list
        dimension = len(img) * len(img[0])
        temp = list(img.reshape(dimension))
        # append label to the end of the list
        temp.append(label)
        output.append(temp)
    return output

class Sigmoid:
    def forward(self, X):
        return 1.0 / (1.0 + np.exp(-X))

    def backward(self, X, dLds):
        s = self.forward(X)
        return (1.0 - s) * s * dLds

class MultiplyGate:
    def forward(self, W, X):
        return np.dot(X, W)

    def backward(self, W, X, dLdZ):
        dLdW = np.dot(np.transpose(X), dLdZ)
        dLdX = np.dot(dLdZ, np.transpose(W))
        return dLdW, dLdX

class AddGate:
    def forward(self, Z, b):
        return Z + b

    def backward(self, Z, b, dLdS):
        dLdZ = dLdS * np.ones_like(Z)
        dLdb = np.dot(np.ones((1, dLdS.shape[0]), dtype=np.float64), dLdS)
        return dLdb, dLdZ

class LSE:
    def eval(self, input):
        return (np.greater(input, 0.5))*1

    def eval_error(self, input, y_true):
        return np.mean(np.power(np.subtract(input,y_true.reshape(-1, 1)), 2))

    def calc_diff(self, input, y_true):
        num_examples = input.shape[0]
        return 2/num_examples * np.subtract(input, y_true.reshape(-1, 1))

class Model:
    def __init__(self, layers_dim):
        self.b = []
        self.W = []
        for i in range(len(layers_dim)-1):
            # initialize weights and biases
            self.W.append(self.randomizer(layers_dim[i], layers_dim[i + 1]))
            self.b.append(self.randomizer(1, layers_dim[i + 1]))

    def randomizer(self, n1, n2):
        return 0.02 * np.random.rand(n1, n2) - 0.01

    def calculate_loss(self, X, y):
        m_Gate = MultiplyGate()
        a_Gate = AddGate()
        layer = Sigmoid()
        output = LSE()

        input = X
        for i in range(len(self.W)):
            mul = m_Gate.forward(self.W[i], input)
            add = a_Gate.forward(mul, self.b[i])
            input = layer.forward(add)
        return output.eval_error(input, y)

    def train(self, X, y, num_epoch, lr, regularization):
        # add gates
        m_Gate = MultiplyGate()
        a_Gate = AddGate()
        layer = Sigmoid()
        output = LSE()

        for epoch in range(num_epoch):
            # Forward propagation
            input = X
            forward = [(None, None, input)]

            # for each layer except the output layer
            for i in range(len(self.W)):
                mul = m_Gate.forward(self.W[i], input)
                add = a_Gate.forward(mul, self.b[i])
                input = layer.forward(add)
                forward.append((mul, add, input))

            # Back propagation

            # derivative of cumulative error from output layer
            dfunc = output.calc_diff(forward[len(forward)-1][2], y)
            for i in range(len(forward)-1, 0, -1):
                dadd = layer.backward(forward[i][1], dfunc)
                db, dmul = a_Gate.backward(forward[i][0], self.b[i-1], dadd)
                dW, dfunc = m_Gate.backward(self.W[i-1], forward[i-1][2], dmul)

                # Add regularization terms
                dW += regularization * self.W[i-1]

                # Update gradient descent parameter
                self.b[i-1] += -lr * db
                self.W[i-1] += -lr * dW

            if epoch % 100 == 0:
                print("Loss after iteration %i: %f" %(epoch, self.calculate_loss(X, y)))

    def predict(self, X):
        m_Gate = MultiplyGate()
        a_Gate = AddGate()
        layer = Sigmoid()

        input = X
        for i in range(len(self.W)):
            mul = m_Gate.forward(self.W[i], input)
            add = a_Gate.forward(mul, self.b[i])
            input = layer.forward(add)

        return (np.greater(input, 0.5)) * 1

def eval(Y_pred, Y_true):
    acc = 0
    for i in range(len(Y_pred)):
        if (int(Y_pred[i]) == Y_true[i]):
            acc = acc + 1
    print(acc * 1.0 / len(Y_true))

if __name__ == '__main__':
    # read file list
    train1, train0 = label(train_data)
    test1, test0 = label(test_data)
    train_image1 = readImage(train1, 1)
    train_image0 = readImage(train0, 0)
    test_image1 = readImage(test1, 1)
    test_image0 = readImage(test0, 0)
    data_train = np.concatenate((train_image1, train_image0), axis=0)
    data_test = np.concatenate((test_image1, test_image0), axis=0)

    # shuffling training data
    np.random.shuffle(data_train)

    # prepare data
    X_train = data_train[:, :-1]
    Y_train = data_train[:, -1]
    X_test = data_test[:, :-1]
    Y_test = data_test[:, -1]

    layers = [np.shape(X_train)[1], num_neurons, 1]

    # BPNN
    model = Model(layers)
    model.train(X_train, Y_train, num_epoch=num_epochs, lr=lr, regularization=0.01)

    # predict train and test data
    y_train_pred = model.predict(X_train)
    y_pred = model.predict(X_test)

    print("training accuracy:")
    eval(y_train_pred, Y_train)

    print("testing accuracy:")
    eval(y_pred, Y_test)
