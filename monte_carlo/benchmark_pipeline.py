import time
import os
import pandas as pd
import numpy as np
from mc_py    import simulate_mc_py
from wrapper  import simulate_mc_cuda
from pipeline import (load_data,
                      filter_by_date_and_region,
                      filter_and_select,
                      add_features,
                      dedupe_and_sort,
                      aggregate_risk,
                      save_outputs)

def full_pipeline_timed(csv_path, base_path,
                        start_year=1991, end_year=2024, uf_allowlist=None,
                        use_cuda=False, mc_N=1_000_000):
    """Executa e mede todas as etapas do pipeline."""
    times = {}
    t0 = time.perf_counter()
    df = load_data(csv_path)
    times['load_data'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = filter_by_date_and_region(df, start_year, end_year, uf_allowlist)
    times['filter_by_date_and_region'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = filter_and_select(df)
    times['filter_and_select'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = add_features(df)
    times['add_features'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df = dedupe_and_sort(df)
    times['dedupe_and_sort'] = time.perf_counter() - t0

    t0 = time.perf_counter()
    df_agg = aggregate_risk(df)
    times['aggregate_risk'] = time.perf_counter() - t0

    # ── Monte Carlo (se quiser incluir no pipeline) ──
    t0 = time.perf_counter()
    if use_cuda:
        mc_res = simulate_mc_cuda(mc_N, mu=0.0, sigma=1.0, seed_offset=0)
    else:
        mc_res = simulate_mc_py(mc_N, mu=0.0, sigma=1.0, seed_offset=0)
    times['monte_carlo'] = time.perf_counter() - t0

    # Salvar vetor MC no disco
    mc_path_csv = os.path.join(base_path, "monte_carlo_results.csv")
    np.savetxt(mc_path_csv, mc_res, delimiter=",")

    t0 = time.perf_counter()
    save_outputs(df, df_agg, base_path)
    times['save_outputs'] = time.perf_counter() - t0


    t0 = time.perf_counter()
    save_outputs(df, df_agg, base_path)
    times['save_outputs'] = time.perf_counter() - t0

    times['total'] = sum(times.values())
    return times

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end do pipeline (I/O → pré-proc → MC → saída)"
    )
    parser.add_argument(
        "--csv", "-c", type=str, required=True,
        help="Caminho para o CSV de entrada (ex.: BD_Atlas_…_Consolidado.csv)"
    )
    parser.add_argument(
        "--out", "-o", type=str, required=True,
        help="Diretório onde salvar os outputs e o relatório"
    )
    parser.add_argument(
        "--cuda", action="store_true",
        help="Use o kernel CUDA em vez da versão Python para Monte Carlo"
    )
    parser.add_argument(
        "--mc_N", type=int, default=1_000_000,
        help="Número de amostras para a simulação Monte Carlo"
    )
    args = parser.parse_args()

    # Executa e mede
    stats = full_pipeline_timed(
        csv_path=args.csv,
        base_path=args.out,
        use_cuda=args.cuda,
        mc_N=args.mc_N
    )

    # Monta o DataFrame de resultados
    df_stats = pd.DataFrame.from_dict(
        stats, orient="index", columns=["tempo_s"]
    )
    df_stats["% do total"] = (
        df_stats["tempo_s"] / df_stats.loc["total", "tempo_s"] * 100
    )

    # Exibe e salva
    print(df_stats)
    report_path = f"{args.out.rstrip('/')}/benchmark_report.csv"
    df_stats.to_csv(report_path)
    print(f"Relatório salvo em: {report_path}")

