"""Executor de experimentos empíricos com LinearSVC, com retomada e exportação para C/C++.

Esta versão segue de forma mais explícita a referência oficial de parâmetros do LinearSVC no scikit-learn:
- dual=False é priorizado porque este projeto possui muito mais amostras do que features.
- As combinações de penalty/loss foram mantidas válidas para LinearSVC/liblinear.
- L1 + squared_hinge foi incluído porque pode gerar coeficientes esparsos, o que é útil para futura inferência em C/C++.
- crammer_singer não foi incluído intencionalmente na grade padrão porque é mais caro e raramente melhora a acurácia na prática.

Casos de uso:
1) Rodada de baseline com todas as features.
2) Rodada futura após feature selection, usando outra pasta de entrada/saída.

Arquivos esperados por padrão:
  ./Data/Xtrain.csv
  ./Data/ytrain.csv
  ./Data/Xtest.csv
  ./Data/ytest.csv

Exemplo de baseline:
  python svm_experiments_cpp_ready.py --data-dir ./Data --output-dir svm_outputs_baseline --tag baseline

Exemplo de rodada com feature selection:
  python svm_experiments_cpp_ready.py --data-dir ./Data_feature_selection --output-dir svm_outputs_fs --tag feature_selection

O script salva os resultados após cada experimento, então é mais seguro deixá-lo rodando por bastante tempo.
Ele também exporta o melhor modelo LinearSVC em JSON para futura inferência em C/C++.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight

RANDOM_STATE = 42

DROP_COLS = ["Unnamed: 0", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p"]
NUMERIC_COLS = [
    "duration",
    "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Executa experimentos empíricos com LinearSVC.")

    parser.add_argument(
        "--data-dir",
        default=str(script_dir.parent / "Data"),
        help="Pasta contendo os arquivos CSV Xtrain/ytrain/Xtest/ytest.",
    )

    parser.add_argument(
        "--output-dir",
        default=str(script_dir / "svm_outputs"),
        help="Pasta onde os resultados serão salvos. Por padrão, fica dentro da pasta SVM.",
    )

    parser.add_argument(
        "--tag",
        default="baseline",
        help="Etiqueta da execução salva nos arquivos de resultado, por exemplo baseline ou feature_selection.",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Ignora experimentos que já existem no CSV de resultados.",
    )

    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Não gera gráficos ROC para economizar tempo.",
    )

    parser.add_argument(
        "--save-all-models",
        action="store_true",
        help="Salva todos os modelos treinados, não apenas o melhor.",
    )

    parser.add_argument(
        "--max-experiments",
        type=int,
        default=None,
        help="Limite opcional para testes rápidos.",
    )

    return parser.parse_args()


def output_paths(output_dir: Path) -> dict[str, Path]:
    return {
        "root": output_dir,
        "reports": output_dir / "reports",
        "models": output_dir / "models",
        "plots": output_dir / "plots",
        "exports": output_dir / "cpp_export",
        "results": output_dir / "svm_experiments_results.csv",
        "sorted": output_dir / "svm_experiments_results_sorted.csv",
    }


def ensure_output_dirs(paths: dict[str, Path]) -> None:
    for key in ["root", "reports", "models", "plots", "exports"]:
        paths[key].mkdir(parents=True, exist_ok=True)


def load_and_preprocess_from_files(data_dir: str):
    data_path = Path(data_dir)
    train_x = data_path / "Xtrain.csv"
    train_y = data_path / "ytrain.csv"
    test_x = data_path / "Xtest.csv"
    test_y = data_path / "ytest.csv"

    X_train_df = pd.read_csv(train_x)
    y_train_df = pd.read_csv(train_y)
    X_test_df = pd.read_csv(test_x)
    y_test_df = pd.read_csv(test_y)

    y_train_raw = y_train_df.iloc[:, 0].values if y_train_df.shape[1] == 1 else y_train_df.values.ravel()
    y_test_raw = y_test_df.iloc[:, 0].values if y_test_df.shape[1] == 1 else y_test_df.values.ravel()

    X_train_df.drop(columns=[c for c in DROP_COLS if c in X_train_df.columns], inplace=True)
    X_test_df.drop(columns=[c for c in DROP_COLS if c in X_test_df.columns], inplace=True)

    # Mantém a ordem das colunas tanto para Python quanto para futura inferência em C/C++.
    feature_names = X_train_df.columns.to_list()

    missing_in_test = [c for c in feature_names if c not in X_test_df.columns]
    extra_in_test = [c for c in X_test_df.columns if c not in feature_names]
    if missing_in_test or extra_in_test:
        raise ValueError(
            "As colunas de features de treino/teste são diferentes. "
            f"Ausentes no teste: {missing_in_test}; extras no teste: {extra_in_test}"
        )
    X_test_df = X_test_df[feature_names]

    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    bool_cols = X_train_df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X_train_df[bool_cols] = X_train_df[bool_cols].astype(np.float32)
        X_test_df[bool_cols] = X_test_df[bool_cols].astype(np.float32)

    scaler = StandardScaler()
    numeric_present = [c for c in NUMERIC_COLS if c in X_train_df.columns]
    if numeric_present:
        X_train_df[numeric_present] = scaler.fit_transform(X_train_df[numeric_present])
        X_test_df[numeric_present] = scaler.transform(X_test_df[numeric_present])

    preprocessing_info = {
        "drop_cols": DROP_COLS,
        "feature_names": feature_names,
        "numeric_present": numeric_present,
        "all_numeric_cols_reference": NUMERIC_COLS,
    }

    return (
        X_train_df.values.astype(np.float32),
        X_test_df.values.astype(np.float32),
        y_train,
        y_test,
        scaler,
        le,
        preprocessing_info,
    )


def compute_clipped_class_weight(y_train: np.ndarray, max_weight: float) -> dict[int, float]:
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weights = np.minimum(weights, max_weight)
    return dict(zip(classes.tolist(), weights.tolist()))


def get_class_weight(y_train: np.ndarray, weight_mode: str | None):
    if weight_mode == "balanced":
        classes = np.unique(y_train)
        weights = compute_class_weight("balanced", classes=classes, y=y_train)
        return dict(zip(classes.tolist(), weights.tolist()))
    if weight_mode == "clipped_20":
        return compute_clipped_class_weight(y_train, max_weight=20)
    if weight_mode == "clipped_50":
        return compute_clipped_class_weight(y_train, max_weight=50)
    if weight_mode == "clipped_100":
        return compute_clipped_class_weight(y_train, max_weight=100)
    return None


def build_svm(
    *,
    C: float,
    class_weight,
    max_iter: int,
    tol: float,
    penalty: str = "l2",
    loss: str = "squared_hinge",
    dual: bool | str = False,
    multi_class: str = "ovr",
    fit_intercept: bool = True,
    intercept_scaling: float = 1.0,
) -> LinearSVC:
    return LinearSVC(
        penalty=penalty,
        loss=loss,
        dual=dual,
        tol=tol,
        C=C,
        multi_class=multi_class,
        fit_intercept=fit_intercept,
        intercept_scaling=intercept_scaling,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        max_iter=max_iter,
    )


def safe_auc(model: LinearSVC, y_test: np.ndarray, scores: np.ndarray) -> float:
    n_classes = len(model.classes_)
    if n_classes == 2:
        return roc_auc_score(y_test, scores)
    y_test_b = label_binarize(y_test, classes=model.classes_)
    return roc_auc_score(y_test_b, scores, multi_class="ovr", average="macro")


def evaluate_model(model: LinearSVC, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, float]:
    preds = model.predict(X_test)
    scores = model.decision_function(X_test)
    return {
        "accuracy": accuracy_score(y_test, preds),
        "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_test, preds, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_test, preds, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, preds, average="weighted", zero_division=0),
        "auc_macro_ovr": safe_auc(model, y_test, scores),
    }


def save_classification_report(model, X_test, y_test, label_encoder, experiment_name: str, reports_dir: Path) -> None:
    preds = model.predict(X_test)
    labels = model.classes_
    target_names = label_encoder.inverse_transform(labels)
    report = classification_report(
        y_test,
        preds,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    (reports_dir / f"classification_report_{experiment_name}.txt").write_text(report, encoding="utf-8")


def save_roc_plot(model, X_test, y_test, experiment_name: str, plots_dir: Path) -> None:
    scores = model.decision_function(X_test)
    n_classes = len(model.classes_)
    plt.figure(figsize=(6, 4))
    if n_classes == 2:
        auc = roc_auc_score(y_test, scores)
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"ROC (AUC = {auc:.4f})")
    else:
        y_test_b = label_binarize(y_test, classes=model.classes_)
        auc = roc_auc_score(y_test_b, scores, multi_class="ovr", average="macro")
        fpr, tpr, _ = roc_curve(y_test_b.ravel(), scores.ravel())
        plt.plot(fpr, tpr, label=f"ROC micro (AUC macro = {auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Taxa de Falsos Positivos")
    plt.ylabel("Taxa de Verdadeiros Positivos")
    plt.title(f"Curva ROC - {experiment_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / f"roc_{experiment_name}.png", dpi=150)
    plt.close()


def exp(
    name: str,
    *,
    C: float = 1.0,
    max_iter: int = 30000,
    tol: float = 1e-4,
    weight_mode: str | None = "balanced",
    penalty: str = "l2",
    loss: str = "squared_hinge",
    dual: bool | str = False,
    multi_class: str = "ovr",
    fit_intercept: bool = True,
    intercept_scaling: float = 1.0,
) -> dict[str, Any]:
    return {
        "name": name,
        "C": C,
        "max_iter": max_iter,
        "tol": tol,
        "weight_mode": weight_mode,
        "penalty": penalty,
        "loss": loss,
        "dual": dual,
        "multi_class": multi_class,
        "fit_intercept": fit_intercept,
        "intercept_scaling": intercept_scaling,
    }


def get_experiments() -> list[dict[str, Any]]:
    return [
        # === SELECIONADOS: MÍNIMO ESSENCIAL (RÁPIDO) ===
        
        # Baseline original já concluído (10k iter, ~5h) - será pulado.
        exp("baseline_C1_iter10000_balanced_auto", C=1.0, max_iter=10000, weight_mode="balanced", dual="auto"),

        # Baseline com max_iter reduzido para ser mais rápido (15k em vez de 30k).
        exp("doc_baseline_C1_iter15000_l2_squared_dualFalse_balanced", C=1.0, max_iter=15000, weight_mode="balanced"),

        # Apenas 2 variações de C importantes para comparação.
        exp("doc_C01_iter15000_l2_squared_dualFalse_balanced", C=0.1, max_iter=15000),
        exp("doc_C10_iter15000_l2_squared_dualFalse_balanced", C=10.0, max_iter=15000),

        # L1 sparse para C/C++ (modelo importante).
        exp("cpp_sparse_C1_iter15000_l1_squared_dualFalse_balanced", C=1.0, max_iter=15000, penalty="l1", loss="squared_hinge", dual=False),
        
        # === COMENTADOS: Adicione conforme necessário ===
        # Experimentos completos (30k+ iter):
        # exp("doc_baseline_C1_iter30000_l2_squared_dualFalse_balanced", C=1.0, max_iter=30000, weight_mode="balanced"),
        # exp("doc_C1_iter50000_l2_squared_dualFalse_balanced", C=1.0, max_iter=50000, weight_mode="balanced"),
        # exp("compare_C1_l2_hinge_dualTrue_balanced", C=1.0, penalty="l2", loss="hinge", dual=True),
        # exp("doc_C1_l2_squared_dualFalse_clipped50", weight_mode="clipped_50"),
        # exp("doc_C001_l2_squared_dualFalse_balanced", C=0.01),
        # exp("doc_C05_l2_squared_dualFalse_balanced", C=0.5),
        # exp("doc_C2_l2_squared_dualFalse_balanced", C=2.0),
    ]


def append_and_save_results(results: list[dict[str, Any]], paths: dict[str, Path]) -> pd.DataFrame:
    results_df = pd.DataFrame(results)
    results_df.to_csv(paths["results"], index=False)
    sorted_df = sort_results(results_df)
    sorted_df.to_csv(paths["sorted"], index=False)
    return sorted_df


def sort_results(results_df: pd.DataFrame) -> pd.DataFrame:
    return results_df.sort_values(
        by=["convergence_warning", "f1_macro", "recall_macro", "accuracy"],
        ascending=[True, False, False, False],
    )


def export_model_for_cpp(
    model: LinearSVC,
    scaler: StandardScaler,
    label_encoder: LabelEncoder,
    preprocessing_info: dict[str, Any],
    experiment: dict[str, Any],
    metrics: dict[str, Any],
    export_path: Path,
) -> None:
    """Exporta informações suficientes para reproduzir a inferência do LinearSVC em C/C++.

    Para LinearSVC multiclasse:
      scores[classe] = soma_i(x_scaled[i] * coef[classe][i]) + intercept[classe]
      predicao = argmax(scores)

    Para LinearSVC binário:
      score = soma_i(x_scaled[i] * coef[0][i]) + intercept[0]
      predicao = classes[1] se score > 0, caso contrário classes[0]
    """
    feature_names = preprocessing_info["feature_names"]
    numeric_present = preprocessing_info["numeric_present"]

    scaler_by_feature: dict[str, dict[str, float]] = {}
    if numeric_present:
        for col, mean, scale in zip(numeric_present, scaler.mean_, scaler.scale_):
            scaler_by_feature[col] = {"mean": float(mean), "scale": float(scale)}

    payload = {
        "model_type": "LinearSVC",
        "sklearn_version": sklearn.__version__,
        "tag": metrics.get("tag"),
        "experiment": experiment,
        "metrics": metrics,
        "feature_order": feature_names,
        "n_features": len(feature_names),
        "numeric_scaled_features": numeric_present,
        "scaler_by_feature": scaler_by_feature,
        "classes_encoded": model.classes_.astype(int).tolist(),
        "classes_original": label_encoder.inverse_transform(model.classes_).tolist(),
        "coef": model.coef_.astype(float).tolist(),
        "intercept": model.intercept_.astype(float).tolist(),
        "binary_rule": "se score > 0 escolha classes[1], caso contrário classes[0]" if len(model.classes_) == 2 else None,
        "multiclass_rule": "escolha a classe com maior score" if len(model.classes_) > 2 else None,
        "notes": [
            "Os valores das features de entrada devem seguir exatamente feature_order.",
            "Aplique a padronização apenas em numeric_scaled_features usando (x - mean) / scale.",
            "Features não listadas em numeric_scaled_features são passadas como valores numéricos já codificados.",
        ],
    }
    export_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def fit_one_experiment(exp, X_train, X_test, y_train, y_test) -> tuple[LinearSVC, dict[str, Any]]:
    class_weight = get_class_weight(y_train, exp["weight_mode"])
    model = build_svm(
        C=exp["C"],
        class_weight=class_weight,
        max_iter=exp["max_iter"],
        tol=exp["tol"],
        penalty=exp.get("penalty", "l2"),
        loss=exp.get("loss", "squared_hinge"),
        dual=exp.get("dual", False),
        multi_class=exp.get("multi_class", "ovr"),
        fit_intercept=exp.get("fit_intercept", True),
        intercept_scaling=exp.get("intercept_scaling", 1.0),
    )

    start_train = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    convergence_warning = any(issubclass(w.category, ConvergenceWarning) for w in caught_warnings)

    start_pred = time.perf_counter()
    metrics = evaluate_model(model, X_test, y_test)
    pred_time = time.perf_counter() - start_pred

    zero_coef_ratio = float((model.coef_ == 0).sum() / model.coef_.size) if model.coef_.size else 0.0

    row = {
        **exp,
        **metrics,
        "coef_zero_ratio": zero_coef_ratio,
        "train_time_seconds": train_time,
        "prediction_time_seconds": pred_time,
        "prediction_time_per_sample_ms": (pred_time / len(y_test)) * 1000,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X_train.shape[1],
        "convergence_warning": convergence_warning,
    }
    return model, row


def load_existing_results(paths: dict[str, Path]) -> list[dict[str, Any]]:
    if not paths["results"].exists():
        return []
    return pd.read_csv(paths["results"]).to_dict("records")


def run_svm_experiments(
    X_train,
    X_test,
    y_train,
    y_test,
    scaler,
    label_encoder,
    preprocessing_info,
    *,
    paths: dict[str, Path],
    tag: str,
    resume: bool,
    no_plots: bool,
    save_all_models: bool,
    max_experiments: int | None,
):
    experiments = get_experiments()
    if max_experiments is not None:
        experiments = experiments[:max_experiments]

    results = load_existing_results(paths) if resume else []
    completed = {row["name"] for row in results} if resume else set()

    for exp in experiments:
        if exp["name"] in completed:
            print(f"\n=== Pulando experimento já concluído: {exp['name']} ===")
            continue

        print(f"\n=== Rodando experimento: {exp['name']} ===")
        model, row = fit_one_experiment(exp, X_train, X_test, y_train, y_test)
        row["tag"] = tag
        results.append(row)

        save_classification_report(model, X_test, y_test, label_encoder, exp["name"], paths["reports"])
        if not no_plots:
            save_roc_plot(model, X_test, y_test, exp["name"], paths["plots"])

        if save_all_models:
            joblib.dump(model, paths["models"] / f"svm_{exp['name']}.joblib")

        append_and_save_results(results, paths)

        print(
            f"accuracy={row['accuracy']:.4f} | "
            f"f1_macro={row['f1_macro']:.4f} | "
            f"recall_macro={row['recall_macro']:.4f} | "
            f"auc_macro_ovr={row['auc_macro_ovr']:.4f} | "
            f"pred_ms/amostra={row['prediction_time_per_sample_ms']:.6f} | "
            f"convergence_warning={row['convergence_warning']}"
        )

    results_df = pd.DataFrame(results)
    sorted_df = sort_results(results_df)
    sorted_df.to_csv(paths["sorted"], index=False)

    best = sorted_df.iloc[0].to_dict()
    best_exp = next(exp for exp in get_experiments() if exp["name"] == best["name"])
    print(f"\nMelhor experimento pelo critério definido: {best['name']}")

    # Retreina o melhor modelo de forma limpa e salva os artefatos para Python e C/C++.
    best_model, best_row = fit_one_experiment(best_exp, X_train, X_test, y_train, y_test)
    best_row["tag"] = tag

    joblib.dump(best_model, paths["models"] / f"best_svm_{best['name']}.joblib")
    joblib.dump(scaler, paths["models"] / "best_svm_scaler.joblib")
    joblib.dump(label_encoder, paths["models"] / "best_svm_label_encoder.joblib")

    pd.Series(preprocessing_info["feature_names"]).to_csv(paths["root"] / "feature_names.csv", index=False, header=["feature"])
    (paths["root"] / "preprocessing_info.json").write_text(
        json.dumps(preprocessing_info, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    export_model_for_cpp(
        best_model,
        scaler,
        label_encoder,
        preprocessing_info,
        best_exp,
        best_row,
        paths["exports"] / f"best_svm_{best['name']}_cpp_export.json",
    )

    print("\nResultados salvos em:")
    print(f"- {paths['results']}")
    print(f"- {paths['sorted']}")
    print(f"- {paths['reports']}/")
    if not no_plots:
        print(f"- {paths['plots']}/")
    print(f"- {paths['models']}/")
    print(f"- {paths['exports']}/")

    cols = [
        "name",
        "accuracy",
        "f1_macro",
        "recall_macro",
        "precision_macro",
        "auc_macro_ovr",
        "prediction_time_per_sample_ms",
        "coef_zero_ratio",
        "n_features",
        "convergence_warning",
    ]
    print("\nRanking resumido:")
    print(sorted_df[cols].to_string(index=False))

    return sorted_df


def main():
    args = parse_args()
    paths = output_paths(Path(args.output_dir))
    ensure_output_dirs(paths)

    print("Carregando dados pre-split e preprocessando...")
    X_train, X_test, y_train, y_test, scaler, le, preprocessing_info = load_and_preprocess_from_files(args.data_dir)
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")
    print(f"Tag da execução: {args.tag}")
    print(f"Features usadas: {len(preprocessing_info['feature_names'])}")

    run_svm_experiments(
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        le,
        preprocessing_info,
        paths=paths,
        tag=args.tag,
        resume=args.resume,
        no_plots=args.no_plots,
        save_all_models=args.save_all_models,
        max_experiments=args.max_experiments,
    )


if __name__ == "__main__":
    main()