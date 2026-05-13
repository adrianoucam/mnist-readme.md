import ctypes
import os

# 1. Carregar a biblioteca compilada
# O abspath pega apenas o nome/caminho do arquivo
caminho_lib = os.path.abspath('minha_lib.dll') 

# O winmode=0 entra aqui, na hora de carregar a DLL!
minha_lib = ctypes.CDLL(caminho_lib, winmode=0)

# ==========================================
# Exemplo 1: Usando a função simples (soma)
# ==========================================
resultado_soma = minha_lib.soma(5, 7)
print(f"Resultado da soma em C: {resultado_soma}")

# ==========================================
# Exemplo 2: Usando arrays (soma_ponderada)
# ==========================================
# É fundamental definir os tipos de entrada e saída quando trabalhamos com floats e ponteiros no C.
minha_lib.soma_ponderada.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_int]
minha_lib.soma_ponderada.restype = ctypes.c_float

# Criando os arrays em Python usando os tipos do C
tamanho = 3
entradas = (ctypes.c_float * tamanho)(1.5, 2.0, -1.0)
pesos = (ctypes.c_float * tamanho)(0.5, 1.0, 0.2)

resultado_array = minha_lib.soma_ponderada(entradas, pesos, tamanho)
print(f"Resultado da soma ponderada em C: {resultado_array}")