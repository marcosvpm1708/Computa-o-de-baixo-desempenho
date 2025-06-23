import time

def tempo_execucao(func):
    def wrapper(self, *args, **kwargs):
        inicio = time.time()
        resultado = func(self, *args, **kwargs)
        fim = time.time()
        print(f"Tempo de execução de {func.__name__}: {fim - inicio:.4f} segundos")
        return resultado
    return wrapper