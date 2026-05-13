# mnist-readme.md
Using MLP in dataset MNIST  - with Core in C and python optimized

Multi-Layer Perceptron (MLP): Core em C & API Híbrida em Python 

Este projeto acadêmico implementa uma rede neural do tipo Multi-Layer Perceptron (MLP) utilizando uma arquitetura híbrida. 

O objetivo é unir a flexibilidade e facilidade de uso do Python (com uma interface que simula o Keras/Scikit-Learn) ao poder de processamento bruto da linguagem C.

Visão Geral do Projeto

Diferente das implementações puramente em Python, este projeto delega todo o treinamento matemático (Forward e Backward Pass) para uma biblioteca dinâmica (.dll) escrita em C. 

Essa abordagem é a mesma utilizada por grandes frameworks como PyTorch e TensorFlow.

Estrutura do Sistema:

Motor Computacional (C): 

Implementação de baixo nível com foco em coerência de cache e vetorização SIMD.

Interface de Alto Nível (Python): Classe Sequential que gerencia matrizes NumPy e faz a ponte via ctypes.

Vantagens da Implementação em CUtilizar C para o treinamento de redes neurais oferece vantagens críticas em contextos acadêmicos e industriais:

Coerência de Cache (Spatial Locality): 

O código foi otimizado para ler matrizes de forma contígua na RAM, minimizando cache misses e mantendo os dados próximos ao processador.

Vetorização SIMD: Ao utilizar flags de compilação modernas, o processador consegue realizar múltiplas operações matemáticas (somas e multiplicações) em um único ciclo de clock.Gerenciamento de Memória In-Place: 

O C manipula os ponteiros de memória alocados pelo NumPy, eliminando o custo de copiar dados entre as linguagens.

Static Inlining: As funções de ativação (como a Sigmoide) são injetadas diretamente nos loops de cálculo, removendo o overhead de chamadas de função.

Compilação e InstalaçãoPara gerar a biblioteca dinâmica no Windows (utilizando MSYS2/MinGW-w64), utilize os comandos abaixo no terminal:Compilação 

Padrão:

Bash

gcc -shared -o mlp_core.dll mlp_core.c

Compilação Otimizada (Alta Performance):

Este comando ativa as otimizações de nível 3 e instrui o compilador a gerar código específico para a sua arquitetura de CPU:

Bash

gcc -shared -O3 -march=native -ffast-math -o mlp_core.dll mlp_core.c

-O3: Ativa otimizações agressivas de velocidade.

-march=native: Habilita instruções modernas (AVX/AVX2) do seu processador.

-ffast-math: Acelera cálculos algébricos permitindo aproximações numéricas rápidas.

Exemplo de Uso
A API foi desenhada para ser intuitiva para quem já utiliza Scikit-Learn ou TensorFlow:

Python

from meu_keras import Sequential, Flatten, Dense

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(256, activation='sigmoid'),
    Dense(10, activation='softmax')
])

model.fit(x_train, y_train, epochs=10)

Explorador de Fluxo de Memória Híbrido

Para entender como os dados viajam entre o Python e o C durante o treinamento, utilize a ferramenta interativa abaixo:

python3 teste_mnist.py

Feature matrix (x_train): (60000, 28, 28)

Target matrix (y_train): (60000,)

Treinamento iniciado. Motor: C nativo (Alta Performance)

Epoch 1/10 - loss: 0.1741 - accuracy: 0.9473 - val_loss: 0.1776 - val_accuracy: 0.9457

Epoch 2/10 - loss: 0.1213 - accuracy: 0.9614 - val_loss: 0.1422 - val_accuracy: 0.9568

Epoch 3/10 - loss: 0.0910 - accuracy: 0.9705 - val_loss: 0.1277 - val_accuracy: 0.9631

Epoch 4/10 - loss: 0.0639 - accuracy: 0.9785 - val_loss: 0.1130 - val_accuracy: 0.9692

Epoch 5/10 - loss: 0.0517 - accuracy: 0.9826 - val_loss: 0.1079 - val_accuracy: 0.9699

Epoch 6/10 - loss: 0.0418 - accuracy: 0.9862 - val_loss: 0.1070 - val_accuracy: 0.9720

Epoch 7/10 - loss: 0.0312 - accuracy: 0.9892 - val_loss: 0.1074 - val_accuracy: 0.9736

Epoch 8/10 - loss: 0.0306 - accuracy: 0.9896 - val_loss: 0.1115 - val_accuracy: 0.9728

Epoch 9/10 - loss: 0.0196 - accuracy: 0.9931 - val_loss: 0.1046 - val_accuracy: 0.9755

Epoch 10/10 - loss: 0.0274 - accuracy: 0.9908 - val_loss: 0.1133 - val_accuracy: 0.9738

Test loss, Test accuracy: [0.10917699358883358, 0.9728]


Licença

Este projeto foi desenvolvido para fins didáticos no estudo das origens do conexionismo e programação de baixo nível. Sinta-se à vontade para clonar e expandir para outras arquiteturas (como CNNs ou RNNs).

