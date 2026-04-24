"""
Detecção de ataques em tráfego de rede IoT usando MLP (Multilayer Perceptron).

Dataset: IoT-23 — registros de conexões de rede rotulados como benignos ou ataques.
Tarefa: classificação binária — prever se uma conexão é um ataque (1) ou benigna (0).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

# --- Configurações globais ---
CSV_PATH = "./iot23_combined.csv"
RANDOM_STATE = 42       # semente para reprodutibilidade
TEST_SIZE = 0.2         # 20% dos dados reservados para teste
BATCH_SIZE = 2048       # número de amostras processadas por passo de gradiente
EPOCHS = 10             # número máximo de épocas de treinamento

# Colunas a remover: índice redundante e IPs (não contribuem para o aprendizado)
DROP_COLS = ["Unnamed: 0", "id.orig_h", "id.resp_h"]
TARGET_COL = "label"    # coluna alvo que será prevista

# Colunas numéricas contínuas que precisam de normalização
NUMERIC_COLS = [
    "id.orig_p", "id.resp_p", "duration",
    "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]


def load_and_preprocess(path: str):
    """
    Carrega o CSV, cria o rótulo binário, divide em treino/teste e normaliza.

    Retorna:
        X_train, X_test  — features normalizadas como arrays float32
        y_train, y_test  — rótulos binários (0=Benign, 1=Ataque)
        scaler           — objeto StandardScaler ajustado no treino (para uso futuro)
    """
    df = pd.read_csv(path)

    # Remove colunas que não agregam informação ao modelo
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True)

    # Converte o rótulo textual em binário: Benign=0, qualquer outro tipo=1 (ataque)
    y = (df[TARGET_COL] != "Benign").astype(np.float32).values
    X = df.drop(columns=[TARGET_COL])

    # Exibe a distribuição das classes para verificar desbalanceamento
    benign = int((y == 0).sum())
    attack = int((y == 1).sum())
    print(f"Distribuição — Benign: {benign:,} | Ataque: {attack:,}")

    # Converte colunas booleanas (ex: proto_tcp, conn_state_SF) para float32
    # pois o TensorFlow não aceita booleanos como entrada
    bool_cols = X.select_dtypes(include="bool").columns
    X[bool_cols] = X[bool_cols].astype(np.float32)

    # Divide em treino e teste mantendo a proporção de classes (stratify)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Normaliza apenas as colunas numéricas contínuas com média=0 e desvio=1.
    # O scaler é ajustado SOMENTE no treino para evitar vazamento de informação do teste.
    scaler = StandardScaler()
    X_train[NUMERIC_COLS] = scaler.fit_transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    return X_train.values.astype(np.float32), X_test.values.astype(np.float32), y_train, y_test, scaler


def build_mlp(input_dim: int) -> keras.Model:
    """
    Constrói e compila a MLP para classificação binária.

    Arquitetura: 3 camadas ocultas (256 → 128 → 64 neurônios) com BatchNorm e Dropout.
    Saída: 1 neurônio com sigmoid — produz a probabilidade de ser um ataque.
    """
    model = keras.Sequential([
        # Camada de entrada: define o número de features que o modelo recebe
        keras.layers.Input(shape=(input_dim,)),

        # Primeira camada oculta: 256 neurônios com ativação ReLU
        keras.layers.Dense(256, activation="relu"),
        # BatchNormalization: estabiliza e acelera o treinamento normalizando as ativações
        keras.layers.BatchNormalization(),
        # Dropout: desliga 30% dos neurônios aleatoriamente por época para evitar overfitting
        keras.layers.Dropout(0.3),

        # Segunda camada oculta: 128 neurônios
        keras.layers.Dense(128, activation="relu"),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        # Terceira camada oculta: 64 neurônios (sem Dropout, mais próximo da saída)
        keras.layers.Dense(64, activation="relu"),

        # Camada de saída: 1 neurônio com sigmoid → valor entre 0 e 1 (probabilidade de ataque)
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        # Adam: otimizador adaptativo
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # Binary crossentropy: função de perda padrão para classificação binária
        loss="binary_crossentropy",
        # Métricas acompanhadas durante o treinamento:
        # - accuracy: percentual de acertos
        # - auc: área sob a curva ROC (quanto mais próximo de 1, melhor)
        # - precision: dos classificados como ataque, quantos realmente são
        # - recall: dos ataques reais, quantos o modelo detectou
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return model


def main():
    print("Carregando e preprocessando dados...")
    X_train, X_test, y_train, y_test, scaler = load_and_preprocess(CSV_PATH)
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape}")

    # Calcula pesos inversamente proporcionais à frequência de cada classe.
    # Isso faz o modelo penalizar mais os erros na classe minoritária (Benign),
    # compensando o desbalanceamento do dataset.
    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print(f"Pesos das classes — Benign: {weights[0]:.3f} | Ataque: {weights[1]:.3f}")

    model = build_mlp(input_dim=X_train.shape[1])
    model.summary()

    callbacks = [
        # Para o treino se a val_loss não melhorar por 3 épocas seguidas
        # e restaura os pesos da melhor época
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        # Reduz o learning rate pela metade se a val_loss estabilizar por 2 épocas
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2),
    ]

    print("\nIniciando treinamento...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),   # avalia no teste ao final de cada época
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
    )

    # Plota a acurácia de treino e teste por época para visualizar o aprendizado
    # e identificar possível overfitting (treino sobe, teste estagna ou cai)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Teste")
    plt.title("Acurácia por época")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=150)
    plt.show()
    print("Gráfico salvo em accuracy_plot.png")

    # Avalia o modelo final no conjunto de teste e exibe todas as métricas
    results = model.evaluate(X_test, y_test, verbose=0)
    names = model.metrics_names
    print("\n--- Resultados no conjunto de teste ---")
    for name, val in zip(names, results):
        print(f"  {name}: {val:.4f}")

    # Salva o modelo treinado em disco no formato Keras nativo
    model.save("mlp_iot23.keras")
    print("\nModelo salvo em mlp_iot23.keras")


if __name__ == "__main__":
    main()
