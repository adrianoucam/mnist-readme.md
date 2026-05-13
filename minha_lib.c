// minha_lib.c
#include <stdio.h>

// Função simples de soma
int soma(int a, int b) {
    return a + b;
}

// Função que processa um array (ex: calcular a soma ponderada de um neurônio)
float soma_ponderada(float* entradas, float* pesos, int tamanho) {
    float resultado = 0.0;
    for (int i = 0; i < tamanho; i++) {
        resultado += entradas[i] * pesos[i];
    }
    return resultado;
}