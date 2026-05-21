import paho.mqtt.client as paho
from paho import mqtt
import struct
import numpy as np
from sklearn.metrics import classification_report
from time import sleep
from threading import Event, Timer

XTEST_PATH = "../Data/Xtest.csv"
YTEST_PATH = "../Data/ytest.csv"

ESP32_FREQ_MHZ = 320
MHZ_TO_HZ_MULT = 1e+6

last_message_event = Event()

def strToBool(str: str):
    return str == "True"

def splitListFormat(list: list):
    types1 = [float] + [float for _ in range(7)] + [strToBool for _ in range(16)]
    types2 = [float] + [int for _ in range(7)] + [int for _ in range(16)]
    
    for i in range(len(types1)):
        list[i] = types1[i](list[i])
        list[i] = types2[i](list[i])

def on_message(client, userdata, msg):
    global ypred, ytest, pred_time, received

    msg = msg.payload.decode("utf-8").split(':')

    ypred.append(int(msg[0]))
    ytest.append(int(msg[1]))
    pred_time.append(int(msg[2]))

    print(len(ypred))

    received += 1
    last_message_event.set()

if __name__ == "__main__":
    labels = ['Attack', 'Benign', 'C&C', 'C&C-FileDownload',
              'C&C-HeartBeat', 'C&C-HeartBeat-FileDownload',
              'C&C-Torii', 'DDoS', 'FileDownload', 'Okiru',
              'PartOfAHorizontalPortScan'
             ]
    
    d = {}

    for i in range(len(labels)):
        d[labels[i]] = i

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

    with open(XTEST_PATH, "r") as xtest_f:
        with open(YTEST_PATH, "r") as ytest_f:
            xtest_f.readline() # consume header
            ytest_f.readline() # consume header

            for xline in xtest_f:
                if sent == cont:
                    break

                yline = ytest_f.readline().strip()
                features = xline.strip().split(',')
                splitListFormat(features)

                pack = struct.pack(f"<f{'i' * 7}{'?' * 16}B", *features, int(d[yline]))

                while sent - received >= 10:
                    sleep(0.1)
                
                client.publish("/esp32/sub", payload=pack, qos=0)
                # sleep(0.01)

                sent += 1
    
    while True:
        last_message_event.clear()

        received = last_message_event.wait(timeout=10)

        if not received:
            break

    client.loop_stop()
    client.disconnect()

    ypred = np.array(ypred)
    ytest = np.array(ytest)
    pred_time = np.array(pred_time)

    print(classification_report(ytest, ypred))
    print()
    avg_pred_ms = (pred_time.mean() / (ESP32_FREQ_MHZ * MHZ_TO_HZ_MULT)) * 1000
    print(f"avg prediction time: {avg_pred_ms} ms")