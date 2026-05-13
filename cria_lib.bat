gcc -shared -o minha_lib.dll minha_lib.c 
gcc -shared -O3 -o mlp_core.dll mlp_core.c 
gcc -shared -O3 -o mlp_mnist.dll mlp_mnist.c

gcc -shared -O3 -march=native -o mlp_mnist.dll mlp_mnist.c

gcc -shared -O3 -march=native -ffast-math -o mlp_core.dll mlp_core.c