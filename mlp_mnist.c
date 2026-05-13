#include <math.h>

// 1. static inline com proteção de memória para evitar NaN (Not a Number)
static inline float sigmoid(float x) {
    if (x < -45.0f) return 0.0f;
    if (x > 45.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

// 2. Uso rigoroso do 'restrict' para garantir Vetorização SIMD pelo processador.
// A assinatura bate exatamente com os argumentos enviados pelo seu Keras falso no Python.
void fit_mnist_c(const float* restrict X, const float* restrict y, 
                 float* restrict W1, float* restrict W2, float* restrict W3, 
                 int n_samples, int n_input, int n_h1, int n_h2, int n_output, 
                 float lr) {

    // Arrays alocados na Stack (VLA do C99), milhares de vezes mais rápido que usar malloc/free
    float A1[n_h1];
    float A2[n_h2];
    float Z3[n_output];
    float A3[n_output];
    
    float d3[n_output];
    float d2[n_h2];
    float d1[n_h1];

    // O C agora itera apenas sobre as amostras. O Python cuida das Épocas!
    for (int i = 0; i < n_samples; i++) {
        
        // Fixa a linha atual da imagem e o rótulo
        const float* restrict sample_X = &X[i * n_input];
        const float* restrict sample_y = &y[i * n_output];

        // ==========================================
        // 1. FORWARD PASS (HPC Cache Coherent)
        // ==========================================
        
        // Camada 1: Entrada -> Oculta 1 (Inversão de Loops)
        for (int j = 0; j < n_h1; j++) A1[j] = 0.0f;
        for (int k = 0; k < n_input; k++) {
            float x_val = sample_X[k]; // Cache escalar
            for (int j = 0; j < n_h1; j++) {
                A1[j] += x_val * W1[k * n_h1 + j]; // Leitura contígua de W1
            }
        }
        for (int j = 0; j < n_h1; j++) A1[j] = sigmoid(A1[j]);

        // Camada 2: Oculta 1 -> Oculta 2 (Inversão de Loops)
        for (int j = 0; j < n_h2; j++) A2[j] = 0.0f;
        for (int k = 0; k < n_h1; k++) {
            float a1_val = A1[k]; // Cache escalar
            for (int j = 0; j < n_h2; j++) {
                A2[j] += a1_val * W2[k * n_h2 + j]; // Leitura contígua de W2
            }
        }
        for (int j = 0; j < n_h2; j++) A2[j] = sigmoid(A2[j]);

        // Camada 3 (Saída): Oculta 2 -> Saída (Sem ativação)
        for (int j = 0; j < n_output; j++) Z3[j] = 0.0f;
        for (int k = 0; k < n_h2; k++) {
            float a2_val = A2[k]; // Cache escalar
            for (int j = 0; j < n_output; j++) {
                Z3[j] += a2_val * W3[k * n_output + j]; // Leitura contígua de W3
            }
        }

        // Ativação Softmax (Convertendo para Probabilidades)
        float max_z = -1e9;
        for (int j = 0; j < n_output; j++) {
            if (Z3[j] > max_z) max_z = Z3[j];
        }
        float exp_sum = 0.0f;
        for (int j = 0; j < n_output; j++) {
            A3[j] = expf(Z3[j] - max_z);
            exp_sum += A3[j];
        }
        for (int j = 0; j < n_output; j++) {
            A3[j] /= exp_sum;
        }

        // ==========================================
        // 2. BACKWARD PASS (Derivadas Sequenciais)
        // ==========================================
        
        // Erro da Camada Final (Simplificação matemática Cross-Entropy + Softmax)
        for (int j = 0; j < n_output; j++) {
            d3[j] = A3[j] - sample_y[j];
        }

        // Retropropagação: Oculta 2 (Estruturada para ler W3 sequencialmente)
        for (int j = 0; j < n_h2; j++) {
            float err = 0.0f;
            for (int k = 0; k < n_output; k++) {
                err += d3[k] * W3[j * n_output + k]; // W3 sendo lido da esquerda p/ direita
            }
            d2[j] = err * A2[j] * (1.0f - A2[j]);
        }

        // Retropropagação: Oculta 1 (Estruturada para ler W2 sequencialmente)
        for (int j = 0; j < n_h1; j++) {
            float err = 0.0f;
            for (int k = 0; k < n_h2; k++) {
                err += d2[k] * W2[j * n_h2 + k]; // W2 sendo lido da esquerda p/ direita
            }
            d1[j] = err * A1[j] * (1.0f - A1[j]);
        }

        // ==========================================
        // 3. ATUALIZAÇÃO DOS PESOS (SIMD Friendly)
        // ==========================================
        
        for (int j = 0; j < n_h2; j++) {
            float a2_val = A2[j]; // Cache
            for (int k = 0; k < n_output; k++) {
                W3[j * n_output + k] -= lr * d3[k] * a2_val;
            }
        }
        
        for (int j = 0; j < n_h1; j++) {
            float a1_val = A1[j]; // Cache
            for (int k = 0; k < n_h2; k++) {
                W2[j * n_h2 + k] -= lr * d2[k] * a1_val;
            }
        }
        
        for (int j = 0; j < n_input; j++) {
            float x_val = sample_X[j]; // Cache
            for (int k = 0; k < n_h1; k++) {
                W1[j * n_h1 + k] -= lr * d1[k] * x_val;
            }
        }
    }
}