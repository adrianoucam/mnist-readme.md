import numpy as np
import ctypes
import os

# Classes falsas para imitar a API do Keras
class Flatten:
    def __init__(self, input_shape=None):
        pass

class Dense:
    def __init__(self, units, activation):
        self.units = units
        self.activation = activation

class History:
    def __init__(self):
        self.history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

class Sequential:
    def __init__(self, layers):
        self.n_input = 784 # 28x28 da imagem
        self.n_h1 = layers[1].units
        self.n_h2 = layers[2].units
        self.n_output = layers[3].units
        
        # Inicialização He (Para evitar que os gradientes sumam no meio da rede)
        self.W1 = (np.random.randn(self.n_input, self.n_h1) * np.sqrt(2./self.n_input)).astype(np.float32)
        self.W2 = (np.random.randn(self.n_h1, self.n_h2) * np.sqrt(2./self.n_h1)).astype(np.float32)
        self.W3 = (np.random.randn(self.n_h2, self.n_output) * np.sqrt(2./self.n_h2)).astype(np.float32)

        # Carrega a nossa biblioteca em C
        caminho_lib = os.path.abspath('mlp_mnist.dll')
        self.lib = ctypes.CDLL(caminho_lib, winmode=0)
        self.lib.fit_mnist_c.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32), np.ctypeslib.ndpointer(dtype=np.float32),
            np.ctypeslib.ndpointer(dtype=np.float32),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
        ]
        
    def compile(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        # Em nossa implementação didática, usaremos SGD no lugar do Adam
        self.lr = 0.05 

    def _forward(self, X):
        # Apenas para a avaliação em Python. O treino pesado está no C.
        def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -45, 45)))
        A1 = sigmoid(np.dot(X, self.W1))
        A2 = sigmoid(np.dot(A1, self.W2))
        Z3 = np.dot(A2, self.W3)
        exp_z = np.exp(Z3 - np.max(Z3, axis=1, keepdims=True))
        A3 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        return A3

    def evaluate(self, X, y, verbose=0):
        if X.ndim > 2: X = X.reshape(X.shape[0], -1) # Faz o trabalho da Camada Flatten
        preds = self._forward(X)
        
        # Transforma o 'y' em one-hot encode se vier como número (ex: 5 -> [0,0,0,0,0,1,0,0,0,0])
        if y.ndim == 1:
            y_onehot = np.zeros((y.size, self.n_output))
            y_onehot[np.arange(y.size), y] = 1.0
        else:
            y_onehot = y
            
        loss = -np.mean(np.sum(y_onehot * np.log(preds + 1e-8), axis=1))
        acc = np.mean(np.argmax(preds, axis=1) == (y if y.ndim == 1 else np.argmax(y, axis=1)))
        return [loss, acc]

    def fit(self, x_train, y_train, epochs=10, batch_size=2000, validation_split=0.2):
        print("Treinamento iniciado. Motor: C nativo (Alta Performance)")
        # A camada Flatten embutida (Transforma a imagem 28x28 numa linha de 784 colunas)
        X = x_train.reshape(x_train.shape[0], -1).astype(np.float32)
        
        y_onehot = np.zeros((y_train.size, self.n_output), dtype=np.float32)
        y_onehot[np.arange(y_train.size), y_train] = 1.0

        val_size = int(X.shape[0] * validation_split)
        train_size = X.shape[0] - val_size

        X_tr, y_tr = X[:train_size], y_onehot[:train_size]
        X_val, y_val = X[train_size:], y_train[train_size:] # mantemos y numérico para avaliação

        X_tr = np.ascontiguousarray(X_tr)
        y_tr = np.ascontiguousarray(y_tr)

        mod = History()

        for ep in range(epochs):
            # Chama o C para processar toda a base de treino!
            self.lib.fit_mnist_c(
                X_tr, y_tr, self.W1, self.W2, self.W3,
                train_size, self.n_input, self.n_h1, self.n_h2, self.n_output, self.lr
            )

            t_loss, t_acc = self.evaluate(X_tr, np.argmax(y_tr, axis=1))
            v_loss, v_acc = self.evaluate(X_val, y_val)

            mod.history['loss'].append(t_loss)
            mod.history['accuracy'].append(t_acc)
            mod.history['val_loss'].append(v_loss)
            mod.history['val_accuracy'].append(v_acc)
            
            print(f"Epoch {ep+1}/{epochs} - loss: {t_loss:.4f} - accuracy: {t_acc:.4f} - val_loss: {v_loss:.4f} - val_accuracy: {v_acc:.4f}")
            
        return mod