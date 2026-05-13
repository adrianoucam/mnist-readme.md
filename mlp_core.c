#include <math.h>

// PRINCÍPIO 1: 'static inline' e Proteção Numérica
// Impede que o processador "salte" na memória para buscar a função, injetando
// a matemática direto no loop. Protege contra explosão de gradiente (NaN).
static inline float sigmoid(float x) {
    if (x < -45.0f) return 0.0f;
    if (x > 45.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

// PRINCÍPIO 2: Ponteiros 'restrict' e 'const'
// A palavra 'restrict' jura para o compilador que X, y, W1 e W2 não se sobrepõem na memória.
// Isso permite que a CPU ative a Vetorização SIMD (calcule 4 a 8 floats por ciclo de clock).
void fit_mlp_c(const float* restrict X, const float* restrict y, 
               float* restrict W1, float* restrict W2, 
               int n_samples, int n_input, int n_hidden, int n_output, 
               float lr, int epochs) {

    // Alocação na Stack (VLA do C99) - Milhares de vezes mais rápido que malloc()
    float hidden_layer[n_hidden];
    float output_layer[n_output];
    float hidden_deltas[n_hidden];
    float output_deltas[n_output];

    for (int ep = 0; ep < epochs; ep++) {
        for (int i = 0; i < n_samples; i++) {
            
            // Garantia de leitura apenas (const) na nossa base de dados
            const float* restrict sample_X = &X[i * n_input];
            const float* restrict sample_y = &y[i * n_output];

            // ==========================================
            // 1. FORWARD PASS (Otimizado para Coerência de Cache)
            // ==========================================
            
            // Zera a camada oculta
            for (int h = 0; h < n_hidden; h++) hidden_layer[h] = 0.0f;
            
            // PRINCÍPIO 3: Inversão de Loops (Spatial Locality)
            // O loop 'in' agora está por FORA. Isso garante que a CPU leia a matriz W1
            // de forma linear (0, 1, 2, 3...) aproveitando 100% da linha de cache!
            for (int in = 0; in < n_input; in++) {
                float x_val = sample_X[in]; // Cache escalar local
                for (int h = 0; h < n_hidden; h++) {
                    hidden_layer[h] += x_val * W1[in * n_hidden + h];
                }
            }
            // Aplica a ativação apenas uma vez no final
            for (int h = 0; h < n_hidden; h++) {
                hidden_layer[h] = sigmoid(hidden_layer[h]);
            }

            // Repetindo a lógica de cache para a camada de saída
            for (int o = 0; o < n_output; o++) output_layer[o] = 0.0f;

            for (int h = 0; h < n_hidden; h++) {
                float h_val = hidden_layer[h];
                for (int o = 0; o < n_output; o++) {
                    output_layer[o] += h_val * W2[h * n_output + o];
                }
            }
            for (int o = 0; o < n_output; o++) {
                output_layer[o] = sigmoid(output_layer[o]);
            }

            // ==========================================
            // 2. BACKWARD PASS 
            // ==========================================
            
            for (int o = 0; o < n_output; o++) {
                float out_val = output_layer[o];
                float erro = sample_y[o] - out_val;
                output_deltas[o] = erro * out_val * (1.0f - out_val);
            }

            for (int h = 0; h < n_hidden; h++) {
                float erro = 0.0f;
                // Leitura já é linear (W2 lido da esquerda para a direita)
                for (int o = 0; o < n_output; o++) {
                    erro += output_deltas[o] * W2[h * n_output + o];
                }
                float h_val = hidden_layer[h];
                hidden_deltas[h] = erro * h_val * (1.0f - h_val);
            }

            // ==========================================
            // 3. AJUSTE DOS PESOS (SIMD Friendly)
            // ==========================================
            
            for (int h = 0; h < n_hidden; h++) {
                float h_val = hidden_layer[h];
                for (int o = 0; o < n_output; o++) {
                    W2[h * n_output + o] += lr * output_deltas[o] * h_val;
                }
            }

            for (int in = 0; in < n_input; in++) {
                float x_val = sample_X[in];
                for (int h = 0; h < n_hidden; h++) {
                    W1[in * n_hidden + h] += lr * hidden_deltas[h] * x_val;
                }
            }
        }
    }
}