import json
import pandas as pd
import os

LOG_PATH = "logs/predict_logs.jsonl"
BASELINE_PATH = "monitoring/baseline_stats.json"

def main():
    if not os.path.exists(LOG_PATH):
        print("Nenhum log encontrado ainda. Use a API primeiro para gerar previsoes.")
        return

    rows = []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line)["input"])
            except Exception:
                continue

    df_prod = pd.DataFrame(rows)

    # 🔧 Corrige tipos: força tudo que for numero a virar numero; invalido vira NaN
    for col in df_prod.columns:
        df_prod[col] = pd.to_numeric(df_prod[col], errors="coerce")

    # Remove linhas totalmente vazias
    df_prod = df_prod.dropna(how="all")

    # Se nao existir baseline ainda, cria
    if not os.path.exists(BASELINE_PATH):
        baseline = df_prod.mean(numeric_only=True).to_dict()
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2, ensure_ascii=False)
        print("Baseline criado com base nos primeiros dados de producao.")
        print("Rode novamente apos mais chamadas para verificar drift.")
        return

    with open(BASELINE_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    print("\nRelatorio simples de drift (media producao vs baseline):")
    for col, base_mean in baseline.items():
        if col in df_prod.columns:
            prod_mean = float(df_prod[col].mean())
            diff = prod_mean - float(base_mean)
            status = "OK"
            if abs(diff) > 0.5:
                status = "POSSIVEL DRIFT"
            print(f"- {col}: baseline={base_mean:.2f} | producao={prod_mean:.2f} | diff={diff:.2f} -> {status}")

if __name__ == "__main__":
    main()
#   python monitoring/drift_check.py