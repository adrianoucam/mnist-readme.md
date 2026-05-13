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


Licença

Este projeto foi desenvolvido para fins didáticos no estudo das origens do conexionismo e programação de baixo nível. Sinta-se à vontade para clonar e expandir para outras arquiteturas (como CNNs ou RNNs).

