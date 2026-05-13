import numpy as np
# Importamos a nossa própria biblioteca como se fosse o Scikit-Learn!
from meu_sklearn import CustomMLPClassifier

# O famoso problema XOR (Ou Exclusivo)
# Saída é 1 apenas se as entradas forem diferentes
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
y = np.array([0, 1, 1, 0])

print("Iniciando treinamento com Custom MLP (C Backend)...")

# Transição transparente para os seus alunos: API idêntica ao Scikit-Learn!
modelo = CustomMLPClassifier(
    hidden_layer_sizes=(4,),  # 4 neurônios na camada oculta
    max_iter=5000,            # 5000 épocas 
    learning_rate_init=0.1    # Taxa de aprendizado
)

modelo.fit(X, y)

print("Treinamento concluído!")

# Fazendo predições
previsoes = modelo.predict(X)

print("\n--- Resultados do Teste XOR ---")
for i in range(len(X)):
    print(f"Entrada: {X[i]} | Esperado: {y[i]} | Rede Neural Previu: {previsoes[i]}")