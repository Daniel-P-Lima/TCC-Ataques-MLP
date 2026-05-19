import paho.mqtt.client as paho
from paho import mqtt
import struct

FILE_PATHNAME = "../Data/Xtest" # Without extension

def strToBool(str: str):
    return str == "True"

def splitListFormat(list: list):
    types1 = [float] + [float for _ in range(7)] + [strToBool for _ in range(16)]
    types2 = [float] + [int for _ in range(7)] + [int for _ in range(16)]
    
    for i in range(len(types1)):
        list[i] = types1[i](list[i])
        list[i] = types2[i](list[i])

if __name__ == "__main__":
    client = paho.Client(client_id="", userdata=None, protocol=paho.MQTTv5)

    client.tls_set(tls_version=mqtt.client.ssl.PROTOCOL_TLS)

    client.username_pw_set("python", "Python123")

    client.connect("8fecfa22d79b48eb9d9e2009dd10c430.s1.eu.hivemq.cloud", 8883)

    client.loop_start()

    with open(f"{FILE_PATHNAME}.csv", "r") as input_file:
        with open(f"{FILE_PATHNAME}.bin", "wb") as output_file:
            input_file.readline() # consume header

            for line in input_file:
                line = line.strip()
                features = line.split(',')
                splitListFormat(features)
                pack = struct.pack(f"<f{'I' * 7}{'?' * 16}", *features)
                client.publish("/esp32/sub", payload=pack, qos=0)

                break
    
    client.loop_stop()
    client.disconnect()