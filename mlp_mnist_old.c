#include <math.h>

// Função Sigmoide com proteção contra estouro de memória (NaN)
float sigmoid(float x) {
    if (x < -45.0f) return 0.0f;
    if (x > 45.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

// O motor de treinamento que processa 1 época inteira
void fit_mnist_c(float* X, float* y, float* W1, float* W2, float* W3, 
                 int n_samples, int n_input, int n_h1, int n_h2, int n_output, 
                 float lr) {

    // Arrays dinâmicos para guardar o sinal passando pelos neurônios (VLA do C99)
    float A1[n_h1];
    float A2[n_h2];
    float Z3[n_output];
    float A3[n_output];
    
    // Arrays para o Backpropagation (Os erros de cada camada)
    float d3[n_output];
    float d2[n_h2];
    float d1[n_h1];

    for (int i = 0; i < n_samples; i++) {
        float* sample_X = &X[i * n_input];
        float* sample_y = &y[i * n_output];

        // ==========================================
        // 1. FORWARD PASS
        // ==========================================
        // Camada 1
        for (int j = 0; j < n_h1; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n_input; k++) sum += sample_X[k] * W1[k * n_h1 + j];
            A1[j] = sigmoid(sum);
        }

        // Camada 2
        for (int j = 0; j < n_h2; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n_h1; k++) sum += A1[k] * W2[k * n_h2 + j];
            A2[j] = sigmoid(sum);
        }

        // Camada de Saída (Sem ativação ainda)
        float max_z = -1e9;
        for (int j = 0; j < n_output; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n_h2; k++) sum += A2[k] * W3[k * n_output + j];
            Z3[j] = sum;
            if (sum > max_z) max_z = sum; // Pega o maior para estabilizar o Softmax
        }

        // Ativação Softmax (Probabilidade)
        float exp_sum = 0.0f;
        for (int j = 0; j < n_output; j++) {
            A3[j] = expf(Z3[j] - max_z);
            exp_sum += A3[j];
        }
        for (int j = 0; j < n_output; j++) {
            A3[j] /= exp_sum;
        }

        // ==========================================
        // 2. BACKWARD PASS (Retropropagação)
        // ==========================================
        
        // A derivada da Cross-Entropy + Softmax simplifica milagrosamente para (Previsão - Real)
        for (int j = 0; j < n_output; j++) {
            d3[j] = A3[j] - sample_y[j]; 
        }

        // Erro da Camada Oculta 2
        for (int j = 0; j < n_h2; j++) {
            float err = 0.0f;
            for (int k = 0; k < n_output; k++) err += d3[k] * W3[j * n_output + k];
            d2[j] = err * A2[j] * (1.0f - A2[j]);
        }

        // Erro da Camada Oculta 1
        for (int j = 0; j < n_h1; j++) {
            float err = 0.0f;
            for (int k = 0; k < n_h2; k++) err += d2[k] * W2[j * n_h2 + k];
            d1[j] = err * A1[j] * (1.0f - A1[j]);
        }

        // ==========================================
        // 3. ATUALIZAÇÃO DOS PESOS (Online SGD)
        // ==========================================
        for (int j = 0; j < n_h2; j++) {
            for (int k = 0; k < n_output; k++) W3[j * n_output + k] -= lr * d3[k] * A2[j];
        }
        for (int j = 0; j < n_h1; j++) {
            for (int k = 0; k < n_h2; k++) W2[j * n_h2 + k] -= lr * d2[k] * A1[j];
        }
        for (int j = 0; j < n_input; j++) {
            for (int k = 0; k < n_h1; k++) W1[j * n_h1 + k] -= lr * d1[k] * sample_X[j];
        }
    }
}