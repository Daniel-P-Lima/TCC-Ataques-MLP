"""
Detecção de ataques em tráfego de rede IoT usando MLP (Multilayer Perceptron).

Dataset: IoT-23 — registros de conexões de rede rotulados como benignos ou ataques.
Tarefa: classificação binária — prever se uma conexão é um ataque (1) ou benigna (0).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras

# --- Configurações globais ---
X_TRAIN_CSV_PATH = "Data/iot23_split/Xtrain.csv"
X_TEST_CSV_PATH = "Data/iot23_split/Xtest.csv"
Y_TRAIN_CSV_PATH = "Data/iot23_split/ytrain.csv"
Y_TEST_CSV_PATH = "Data/iot23_split/ytest.csv"

RANDOM_STATE = 42  # semente para reprodutibilidade
BATCH_SIZE = 2048  # número de amostras processadas por passo de gradiente
EPOCHS = 40  # número máximo de épocas de treinamento


def load_and_preprocess():
    """
    Carrega o CSV, cria o rótulo binário, divide em treino/teste e normaliza.

    Retorna:
        X_train, X_test  — features normalizadas como arrays float32
        y_train, y_test  — labels
    """
    X_train = pd.read_csv(X_TRAIN_CSV_PATH)
    X_test = pd.read_csv(X_TEST_CSV_PATH)
    y_train = pd.read_csv(Y_TRAIN_CSV_PATH)
    y_test = pd.read_csv(Y_TEST_CSV_PATH)

    y_train = y_train.values.ravel() # Compacta o array de dimensão 1-D
    y_test = y_test.values.ravel() # Compacta o array de dimensão 1-D

    # Codifica rótulos string ("Benign", "Attack", ...) para inteiros (0, 1, ...)
    le = LabelEncoder()
    y_train = le.fit_transform(y_train).astype(np.int32)
    y_test = le.transform(y_test).astype(np.int32)
    print(f"Classes codificadas: {list(le.classes_)} → dtype: {y_train.dtype}")
    return X_train.values.astype(np.float32), X_test.values.astype(np.float32), y_train, y_test


def build_mlp(input_dim: int, num_classes: int) -> keras.Model:
    """
    Constrói e compila a MLP para classificação binária.

    Arquitetura: 3 camadas ocultas (256 → 128 → 64 neurônios) com BatchNorm e Dropout.
    Saída: 11 neurônios com softmax — mostra qual ataque pode ser.
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

        # Camada de saída: 11 neurônios com softmax → valor entre 0 e 10 (probabilidade de qual ataque ser)
        keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        # Adam: otimizador adaptativo
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        # Binary crossentropy: função de perda padrão para classificação binária
        loss="sparse_categorical_crossentropy",
        # Métricas acompanhadas durante o treinamento:
        # - accuracy: percentual de acertos
        metrics=["accuracy"]
    )
    return model


def convert_to_tflite(model: keras.Model, output_path: str = "mlp_iot23.tflite"):
    """
    Converte o modelo Keras para o formato TensorFlow Lite (.tflite).
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Quantização dinâmica: converte pesos de float32 para int8 em tempo de conversão.
    # Reduz o tamanho do modelo em ~4x e acelera a inferência na maioria dos dispositivos.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(output_path, "wb") as f:
        f.write(tflite_model)

    original_kb = model.count_params() * 4 / 1024
    tflite_kb = len(tflite_model) / 1024
    print(f"\nModelo TFLite salvo em {output_path}")
    print(f"  Tamanho estimado Keras:  {original_kb:.1f} KB")
    print(f"  Tamanho TFLite (int8):   {tflite_kb:.1f} KB")


def main():
    print("Carregando e preprocessando dados...")
    X_train, X_test, y_train, y_test = load_and_preprocess()

    unique_classes = np.unique(y_train)
    num_classes = len(unique_classes)
    print(f"Treino: {X_train.shape} | Teste: {X_test.shape} | Classes detectadas: {num_classes}")

    # Calcula pesos inversamente proporcionais à frequência de cada classe.
    # Isso faz o modelo penalizar mais os erros na classe minoritária (Benign),
    # compensando o desbalanceamento do dataset.
    weights = compute_class_weight("balanced", classes=unique_classes, y=y_train)
    class_weight_dict = {i: weights[i] for i in range(num_classes)}

    model = build_mlp(input_dim=X_train.shape[1], num_classes=num_classes)
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
        validation_data=(X_test, y_test),  # avalia no teste ao final de cada época
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
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
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia final no Teste: {acc:.4f}")

    # Salva o modelo treinado em disco no formato Keras nativo
    model.save("mlp_iot23.keras")
    print("\nModelo salvo em mlp_iot23.keras")

    convert_to_tflite(model)


if __name__ == "__main__":
    main()
