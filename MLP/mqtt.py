import paho.mqtt.client as paho
from paho import mqtt
import struct
import numpy as np
from sklearn.metrics import classification_report
from time import sleep
from threading import Event

XTEST_PATH = "../Data/iot23_std/Xtest.csv"
YTEST_PATH = "../Data/iot23_std/ytest.csv"

ESP32_FREQ_MHZ = 240  # ESP32 classic (verify with esp_clk_cpu_freq() if needed)
MHZ_TO_HZ_MULT = 1e+6

# 24 float32 features + 1 uint8 label = 97 bytes
# Matches MlpMqttMessage on the ESP32 side
PACK_FMT = '<' + 'f' * 24 + 'B'

LABELS = [
    'Attack', 'Benign', 'C&C', 'C&C-FileDownload',
    'C&C-HeartBeat', 'C&C-HeartBeat-FileDownload',
    'C&C-Torii', 'DDoS', 'FileDownload', 'Okiru',
    'PartOfAHorizontalPortScan'
]
label_to_idx = {lbl: i for i, lbl in enumerate(LABELS)}

last_message_event = Event()


def on_message(client, userdata, msg):
    global ypred, ytest, pred_time, received

    parts = msg.payload.decode("utf-8").split(':')
    ypred.append(int(parts[0]))
    ytest.append(int(parts[1]))
    pred_time.append(int(parts[2]))

    print(len(ypred))

    received += 1
    last_message_event.set()


if __name__ == "__main__":
    client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
    client.on_message = on_message

    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)
    client.username_pw_set("python", "Python123")
    client.connect("8fecfa22d79b48eb9d9e2009dd10c430.s1.eu.hivemq.cloud", 8883)
    client.subscribe("/esp32/pub")
    client.loop_start()

    ypred = []
    ytest = []
    pred_time = []
    cont = 1000

    sent = 0
    received = 0

    with open(XTEST_PATH, "r") as xtest_f, open(YTEST_PATH, "r") as ytest_f:
        xtest_f.readline()  # consume header
        ytest_f.readline()  # consume header

        for xline in xtest_f:
            if sent == cont:
                break

            yline = ytest_f.readline().strip()
            features = [float(v) for v in xline.strip().split(',')]

            pack = struct.pack(PACK_FMT, *features, label_to_idx[yline])

            while sent - received >= 10:
                sleep(0.1)

            client.publish("/esp32/sub", payload=pack, qos=0)
            sent += 1

    while True:
        last_message_event.clear()
        got_message = last_message_event.wait(timeout=10)
        if not got_message:
            break

    client.loop_stop()
    client.disconnect()

    ypred = np.array(ypred)
    ytest = np.array(ytest)
    pred_time = np.array(pred_time)

    print(classification_report(ytest, ypred, target_names=LABELS))
    avg_ms = (pred_time.mean() / (ESP32_FREQ_MHZ * MHZ_TO_HZ_MULT)) * 1000
    print(f"avg prediction time: {avg_ms:.4f} ms")
