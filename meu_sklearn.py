import numpy as np
import ctypes
import os

class CustomMLPClassifier:
    """
    Uma réplica didática do sklearn.neural_network.MLPClassifier
    que usa C para processamento pesado e NumPy para matrizes.
    """
    def __init__(self, hidden_layer_sizes=(5,), max_iter=200, learning_rate_init=0.01):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate_init = learning_rate_init
        self.coefs_ = [] # Guardará [W1, W2] idêntico ao Scikit-Learn
        
        # 1. Carrega a Biblioteca C
        caminho_lib = os.path.abspath('mlp_core.dll')
        self._lib = ctypes.CDLL(caminho_lib, winmode=0)
        
        # 2. Define o "Contrato" de Tipos (A ponte NumPy -> C)
        self._lib.fit_mlp_c.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # X
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # y
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # W1
            np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'), # W2
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int
        ]

    def fit(self, X, y):
        # Garante que os dados são float32 contíguos na memória (C-Style)
        X = np.ascontiguousarray(X, dtype=np.float32)
        if y.ndim == 1:
            y = y.reshape(-1, 1) # Transforma [1, 0] em [[1], [0]]
        y = np.ascontiguousarray(y, dtype=np.float32)

        n_samples, n_input = X.shape
        n_output = y.shape[1]
        n_hidden = self.hidden_layer_sizes[0]

        # NumPy aloca a memória e inicializa os pesos com valores aleatórios pequenos
        W1 = np.random.randn(n_input, n_hidden).astype(np.float32) * 0.1
        W2 = np.random.randn(n_hidden, n_output).astype(np.float32) * 0.1

        # Mágica: Enviamos os ponteiros para o C. 
        # O C vai modificar W1 e W2 diretamente na RAM!
        self._lib.fit_mlp_c(
            X, y, W1, W2,
            n_samples, n_input, n_hidden, n_output,
            self.learning_rate_init, self.max_iter
        )

        self.coefs_ = [W1, W2] # Salva na classe
        return self

    def predict(self, X):
        """ O predict pode ser feito todo em NumPy usando as multiplicações nativas """
        X = np.ascontiguousarray(X, dtype=np.float32)
        W1, W2 = self.coefs_

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        # Forward pass linear algébrico (muito elegante em Python)
        hidden = sigmoid(np.dot(X, W1))
        output = sigmoid(np.dot(hidden, W2))

        # Classificação binária: se a probabilidade for > 0.5, é classe 1
        return (output > 0.5).astype(int).flatten()