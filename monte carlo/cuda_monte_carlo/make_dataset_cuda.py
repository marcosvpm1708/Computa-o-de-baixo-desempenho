# make_dataset_cuda_full.py

import os
import pandas as pd
from wrapper import simulate_mc_cuda

# 1) Parâmetros
INPUT_CSV = "../dataframe/BD_Atlas_1991_2024_v1.0_2025.04.14_Consolidado.csv"
OUTPUT_DIR = "output_dir"
FINAL_CSV = os.path.join(OUTPUT_DIR, "dataset_montecarlo_CUDA_full.csv")
MU, SIGMA, SEED = 0.0, 1.0, 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) Carrega **tudo** com sep=';'
df = pd.read_csv(
    INPUT_CSV,
    sep=';',
    dtype=str,
    encoding='latin1',
    engine='python',
    on_bad_lines='skip'
)

# 3) Normalize colunas de data (sem descartar nada)
df.columns = df.columns.str.strip().str.replace('\ufeff','')
if 'Data_Registro' in df.columns:
    df['Data_Registro'] = pd.to_datetime(df['Data_Registro'],
                                         dayfirst=True, errors='coerce')
elif 'Data Registro' in df.columns:
    df.rename(columns={'Data Registro':'Data_Registro'}, inplace=True)
    df['Data_Registro'] = pd.to_datetime(df['Data_Registro'],
                                         dayfirst=True, errors='coerce')
# mesmo para Data_Evento
if 'Data_Evento' in df.columns:
    df['Data_Evento'] = pd.to_datetime(df['Data_Evento'],
                                       dayfirst=True, errors='coerce')
elif 'Data Evento' in df.columns:
    df.rename(columns={'Data Evento':'Data_Evento'}, inplace=True)
    df['Data_Evento'] = pd.to_datetime(df['Data_Evento'],
                                       dayfirst=True, errors='coerce')

# 4) Gere N = n_linhas amostras Monte Carlo em CUDA
N = len(df)
print(f"Gerando {N} amostras Monte Carlo em CUDA…")
mc = simulate_mc_cuda(N, MU, SIGMA, SEED)

# 5) Anexe como coluna (matching linha a linha)
df['mc_simulation'] = mc

# 6) Salve com sep=';' para ter mesma estrutura BR
df.to_csv(FINAL_CSV, index=False, sep=';')
print(f"Dataset CUDA full salvo em: {FINAL_CSV}")
