# pipeline.py

import os
import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=';',
        dtype=str,
        encoding='latin1',
        engine='python',
        on_bad_lines='skip'
    )
    # resto do código de normalização...

    # DEBUG: listar exatamente as colunas carregadas
    print("\n>> Colunas lidas em load_data:", df.columns.tolist(), "\n")
    # ... resto do código ...


    # Normaliza nomes de colunas: remove espaços e BOM
    df.columns = df.columns.str.strip().str.replace('﻿', '')

    # Mapeia variações comuns
    rename_map = {
        'Data Registro':  'Data_Registro',
        'Data Registro ': 'Data_Registro',
        'Data Evento':    'Data_Evento',
        'Data Evento ':   'Data_Evento',
    }
    df = df.rename(columns=rename_map)

    # Fallback automático para colunas de data
    if 'Data_Registro' not in df.columns:
        candidates = [c for c in df.columns if 'Registro' in c]
        if candidates:
            df = df.rename(columns={candidates[0]: 'Data_Registro'})
    if 'Data_Evento' not in df.columns:
        candidates = [c for c in df.columns if 'Evento' in c]
        if candidates:
            df = df.rename(columns={candidates[0]: 'Data_Evento'})

    # Converte colunas de data
    if 'Data_Registro' in df.columns:
        df['Data_Registro'] = pd.to_datetime(
            df['Data_Registro'], dayfirst=True, errors='coerce'
        )
    else:
        raise KeyError("Coluna 'Data_Registro' não encontrada após carregamento e normalização.")
    if 'Data_Evento' in df.columns:
        df['Data_Evento'] = pd.to_datetime(
            df['Data_Evento'], dayfirst=True, errors='coerce'
        )
    else:
        raise KeyError("Coluna 'Data_Evento' não encontrada após carregamento e normalização.")

    return df


def filter_by_date_and_region(df: pd.DataFrame,
                              start_year=1991,
                              end_year=2024,
                              uf_allowlist=None) -> pd.DataFrame:
    mask_date = df['Data_Evento'].dt.year.between(start_year, end_year)
    df = df.loc[mask_date]
    if uf_allowlist:
        df = df[df['Sigla_UF'].isin(uf_allowlist)]
    return df


def filter_and_select(df: pd.DataFrame) -> pd.DataFrame:
    keys = [
        'Protocolo_S2iD', 'Cod_Cobrade', 'Cod_IBGE_Mun',
        'Nome_Municipio', 'Sigla_UF', 'regiao',
        'Data_Registro', 'Data_Evento'
    ]
    df_sel = df.loc[:, keys].copy()
    df_sel[['Nome_Municipio','Sigla_UF','regiao']] = (
        df_sel[['Nome_Municipio','Sigla_UF','regiao']]
          .fillna('')
    )
    df_sel['Cod_IBGE_Mun'] = pd.to_numeric(
        df_sel['Cod_IBGE_Mun'], errors='coerce'
    ).fillna(0).astype(int)
    return df_sel


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df['ano_evento']   = df['Data_Evento'].dt.year
    df['mes_evento']   = df['Data_Evento'].dt.month
    df['risk_count']   = 1
    return df


def dedupe_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates(subset=['Protocolo_S2iD'])
    return df.sort_values(
        ['Data_Evento','Nome_Municipio']
    ).reset_index(drop=True)


def aggregate_risk(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df.groupby(['Nome_Municipio','ano_evento'])
          .agg(total_eventos=('risk_count','sum'))
          .reset_index()
    )
    return agg


def save_outputs(df_raw: pd.DataFrame,
                 df_agg: pd.DataFrame,
                 base_path: str) -> None:
    os.makedirs(base_path, exist_ok=True)
    df_raw.to_csv(
        os.path.join(base_path, "processed_events.csv"),
        index=False
    )
    df_raw.to_parquet(
        os.path.join(base_path, "processed_events.parquet"),
        index=False
    )
    df_agg.to_csv(
        os.path.join(base_path, "aggregated_risk.csv"),
        index=False
    )
    df_agg.to_parquet(
        os.path.join(base_path, "aggregated_risk.parquet"),
        index=False
    )
