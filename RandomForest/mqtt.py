import paho.mqtt.client as paho
from paho import mqtt
import struct
import numpy as np
from sklearn.metrics import classification_report
from time import sleep
import pandas as pd
from RepeatingTimer import RepeatingTimer

XTEST_PATH = "../Data/Xtest.csv"
YTEST_PATH = "../Data/ytest.csv"

ESP32_FREQ_MHZ = 320
MHZ_TO_HZ_MULT = 1e+6

ANSWER_TIMEOUT_S = 5.0

# last_message_event = Event()

def strToBool(str: str):
    return str == "True"

def splitListFormat(list: list):
    types1 = [float] + [float for _ in range(7)] + [strToBool for _ in range(16)]
    types2 = [float] + [int for _ in range(7)] + [int for _ in range(16)]
    
    for i in range(len(types1)):
        list[i] = types1[i](list[i])
        list[i] = types2[i](list[i])

def resend(pack:bytes):
    client.publish("/esp32/sub", payload=pack, qos=0)

def on_message(client, userdata, msg:paho.MQTTMessage):
    global ypred, ytest, pred_time, sent

    msg_l = msg.payload.decode("utf-8").split(':')

    try:
        id = int(msg_l[3])

        i = next(i for i, t in enumerate(sent) if t[0] == id)
        sent[i][1].cancel()
        del sent[i]

        ypred.append(int(msg_l[0]))
        ytest.append(int(msg_l[1]))
        pred_time.append(int(msg_l[2]))
    except StopIteration:
        pass

    print(len(ypred))

if __name__ == "__main__":
    labels = [
              'Attack', 'Benign', 'C&C', 'C&C-FileDownload',
              'C&C-HeartBeat', 'C&C-HeartBeat-FileDownload',
              'C&C-Torii', 'DDoS', 'FileDownload', 'Okiru',
              'PartOfAHorizontalPortScan'
             ]
    
    d = {}

    for i in range(len(labels)):
        d[labels[i]] = i

    # client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)
    client = paho.Client(callback_api_version=paho.CallbackAPIVersion.VERSION2)
    client.on_message = on_message

    # client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)

    # client.username_pw_set("python", "Python123")

    client.connect("localhost", 1883)
    client.subscribe("/esp32/pub")

    client.loop_start()

    ypred = []
    ytest = []
    pred_time = []

    sent = []

    with open(XTEST_PATH, "r") as xtest_f:
        with open(YTEST_PATH, "r") as ytest_f:
            xtest_f.readline() # consume header
            ytest_f.readline() # consume header
            i = 0

            for xline in xtest_f:
                yline = ytest_f.readline().strip()
                features = xline.strip().split(',')
                splitListFormat(features)

                pack = struct.pack(f"<f{'i' * 7}{'?' * 16}Bi", *features, int(d[yline]), i)

                while len(sent) > 10:
                    sleep(0.1)
                
                client.publish("/esp32/sub", payload=pack, qos=0)
                sleep(0.01)

                timer = RepeatingTimer(ANSWER_TIMEOUT_S, resend, args=[pack])
                sent.append((i, timer))
                timer.start()
                i += 1
    
    while sent:
        sleep(1)

    client.loop_stop()
    client.disconnect()

    ypred = np.array(ypred)
    ytest = np.array(ytest)
    pred_time = np.array(pred_time)

    pd.DataFrame(ypred).to_csv("../Data/results/ypred.csv", index=False)
    pd.DataFrame(ytest).to_csv("../Data/results/ytest.csv", index=False)

    print(classification_report(ytest, ypred))
    print()
    avg_pred_ms = (pred_time.mean() / (ESP32_FREQ_MHZ * MHZ_TO_HZ_MULT)) * 1000
    print(f"avg prediction time: {avg_pred_ms} ms")