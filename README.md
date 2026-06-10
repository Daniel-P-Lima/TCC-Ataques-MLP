# Detecção de Ataques em Tráfego de Rede IoT

Trabalho de Conclusão de Curso — classificação de malware em tráfego de rede IoT.

---

## Dataset

**IoT-23** — dataset público de tráfego de rede capturado em dispositivos IoT reais, contendo conexões rotuladas como benignas ou ataques de diferentes categorias (DDoS, Port Scan, C&C, Okiru, etc.).

Arquivo esperado: `csv/iot23_combined.csv`

| Coluna | Descrição |
|---|---|
| `id.orig_h` / `id.resp_h` | IPs de origem e destino (removidos no pré-processamento) |
| `duration` | Duração da conexão em segundos |
| `orig_bytes` / `resp_bytes` | Bytes enviados/recebidos |
| `orig_pkts` / `resp_pkts` | Pacotes enviados/recebidos |
| `missed_bytes` | Bytes perdidos durante a captura |
| `proto_tcp`, `proto_udp`, `proto_icmp` | Protocolo utilizado (one-hot) |
| `conn_state_*` | Estado da conexão (one-hot: SF, S0, REJ, etc.) |
| `label` | **Alvo**: `Benign` ou tipo de ataque |

Distribuição aproximada:

```
PartOfAHorizontalPortScan    825.939
Okiru                        262.690
Benign                       197.809
DDoS                         138.777
C&C                           15.100
...
```

---

## Instalação

Requer Python 3.9+.

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

---

## Como Rodar

<details>
    <summary>MLP</summary>

```bash
python3 main.py
```

O script executa automaticamente todas as etapas abaixo em sequência.

---

## Pipeline

### 1. Pré-processamento (`load_and_preprocess`)

- Carrega o CSV com `pandas`
- Remove colunas sem valor preditivo: índice, IPs e portas
- Converte o rótulo para binário: `Benign → 0`, qualquer ataque → `1`
- Converte colunas booleanas para `float32` (requisito do TensorFlow)
- Divide os dados em **80% treino / 20% teste** com split estratificado (mantém proporção das classes)
- Aplica `StandardScaler` nas colunas numéricas contínuas — ajustado **apenas no treino** para evitar vazamento de informação

### 2. Arquitetura da MLP (`build_mlp`)

```
Entrada (N features)
    ↓
Dense(256, relu) → BatchNorm → Dropout(0.3)
    ↓
Dense(128, relu) → BatchNorm → Dropout(0.3)
    ↓
Dense(64, relu)
    ↓
Dense(1, sigmoid)  →  probabilidade de ser ataque [0, 1]
```

- **BatchNormalization**: estabiliza e acelera o treinamento normalizando as ativações entre camadas
- **Dropout(0.3)**: desativa 30% dos neurônios aleatoriamente por época para evitar overfitting
- **Sigmoid**: saída entre 0 e 1, interpretada como probabilidade de ataque

Compilado com:
- Otimizador: `Adam` (learning rate = 0.001)
- Loss: `binary_crossentropy`
- Métricas: `accuracy`, `AUC`, `Precision`, `Recall`

### 3. Treinamento

- Até 40 épocas com `batch_size = 2048`
- Usa `class_weight` balanceado para compensar o desbalanceamento do dataset (a classe majoritária recebe peso menor)
- Avalia no conjunto de teste ao final de cada época (`validation_data`)

### 4. Avaliação e Gráfico

Após o treinamento, exibe as métricas finais no conjunto de teste e salva o gráfico de acurácia por época em `accuracy_plot.png`.

> **Por que não usar só acurácia?**
> O dataset é desbalanceado — um modelo que classifica tudo como ataque teria ~86% de acurácia sem aprender nada. Por isso, **Precision** e **Recall** são as métricas mais relevantes.

### 5. Exportação

Dois arquivos são gerados:

| Arquivo | Uso |
|---|---|
| `mlp_iot23.keras` | Recarregar, continuar treinando ou usar em servidor |
| `mlp_iot23.tflite` | Deploy em dispositivos edge (Raspberry Pi, gateway IoT) |

A conversão TFLite aplica **quantização dinâmica** (pesos `float32 → int8`), reduzindo o tamanho do modelo em ~4x.

---

## Recarregar o Modelo

```python
from tensorflow import keras

model = keras.models.load_model("mlp_iot23.keras")
predictions = model.predict(X_new)  # retorna probabilidade [0, 1]
labels = (predictions > 0.5).astype(int)  # 0=Benign, 1=Ataque
```

</details>

<details>
    <summary>Random Forest</summary>

### Pré-Requisitos
ESP-IDF (https://github.com/espressif/esp-idf)
Mosquitto (https://mosquitto.org/download/)

### 1. Dataset

- Faça o download dos arquivos da base IoT23 em https://www.stratosphereips.org/datasets-iot23 (lighter version) e extraia os conteúdos da pasta IoTScenarios na pasta /Preprocessing/IoTScenarios do repositório (mesmo o dataset pré-processado era muito grande para manter no repositório)
- Execute o código /Preprocesing/preprocessing.py de dentro da pasta /Preprocessing/
- Os csvs de treino e teste deverão estar na pasta /Data/ após isso

### 2. Geração das Árvores para Embarque

- Execute o código /RandomForest/RF.py de dentro da pasta /RandomForest/
- A pasta /RandomForest/binTrees/ deve ser preenchida com 26 arquivos binários

### 3. Embarque do Modelo

- É recomendado copiar toda a pasta /RandomForest/mqtt/ para o caminho /<pasta de instalação do esp-idf>/examples/protocols/mqtt/, pois a ferramenta de build, por vezes, não encontra o caminho para dependências
- Garanta que o arquivo de configuração do Mosquitto possua "listener 1883" e "allow_anonymous true" descomentados
- Execute o Mosquitto em um dispositivo na rede local
- Altere o valor da constante ADDR em /<pasta de instalação do esp-idf>/examples/protocols/mqtt/main/app_main.c para o endereço do dispositivo rodando o Mosquitto
- Execute o `.\export.bat` (Windows) ou `export.sh` (Linux) no terminal na pasta de instalação do esp-idf
- Navegue, ainda no terminal, para a pasta /<pasta de instalação do esp-idf>/examples/protocols/mqtt/
- Execute `idf.py menuconfig` e configure as opções abaixo:
- Serial flasher config -> Flash size (4 MB)
- Partition Table -> selecione partitions.csv
- ESP-TLS -> marque Allow potential insecure option, e Skip server certificate verification by default
- Configure conexão Wi-Fi em Example connection configuration
- Salve as configurações e saia do menuconfig
- Execute `idf.py build`
- Conecte a ESP32 ao computador e identifique a porta de conexão
- Execute `idf.py -p <PORTA> flash monitor`

### 4. Validação
- Execute o código /RandomForest/mqtt.py de dentro da pasta /RandomForest
- No caso de a ESP32 desconectar durante o processo, apenas realize o flash do zero no esp-idf e o código Python irá retransmitir as mensagens não respondidas
</details>
