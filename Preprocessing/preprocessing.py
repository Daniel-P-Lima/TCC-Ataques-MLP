import pandas as pd
import ipaddress

def isIPv4(ip_str: str):
    return ipaddress.ip_address(ip_str).version == 4

def isIPv6(ip_str: str):
    return ipaddress.ip_address(ip_str).version == 6

def ipStrToInt(ip_str: str):
    return int(ipaddress.ip_address(ip_str))

def ipStrToIntHigh(ip_str: str):
    return ipStrToInt(ip_str) >> 64

def ipStrToIntLow(ip_str: str):
    return ipStrToInt(ip_str) & ((1 << 64) - 1)

# Files in IoTScenarios can be acquired in https://www.stratosphereips.org/datasets-iot23 (lighter version of the dataset)

captures = [
    "./IoTScenarios/CTU-IoT-Malware-Capture-34-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-43-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-44-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-49-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-52-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-20-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-21-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-42-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-60-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-17-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-36-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-33-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-8-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-35-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-48-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-39-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-7-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-9-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-3-1/bro/conn.log.labeled",
    "./IoTScenarios/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled"
]

frames = []

for capture in captures:
    df = pd.read_table(filepath_or_buffer=capture, skiprows=10, nrows=100000)

    df.columns=['ts',
                'uid',
                'id.orig_h',
                'id.orig_p',
                'id.resp_h',
                'id.resp_p',
                'proto',
                'service',
                'duration',
                'orig_bytes',
                'resp_bytes',
                'conn_state',
                'local_orig',
                'local_resp',
                'missed_bytes',
                'history',
                'orig_pkts',
                'orig_ip_bytes',
                'resp_pkts',
                'resp_ip_bytes',
                'label']
    
    df.drop(df.tail(1).index, inplace=True)

    frames.append(df)

df = pd.concat(frames)

df.loc[(df.label == '-   Malicious   PartOfAHorizontalPortScan'), 'label'] = 'PartOfAHorizontalPortScan'
df.loc[(df.label == '(empty)   Malicious   PartOfAHorizontalPortScan'), 'label'] = 'PartOfAHorizontalPortScan'
df.loc[(df.label == '-   Malicious   Okiru'), 'label'] = 'Okiru'
df.loc[(df.label == '(empty)   Malicious   Okiru'), 'label'] = 'Okiru'
df.loc[(df.label == '-   Benign   -'), 'label'] = 'Benign'
df.loc[(df.label == '(empty)   Benign   -'), 'label'] = 'Benign'
df.loc[(df.label == '-   Malicious   DDoS'), 'label'] = 'DDoS'
df.loc[(df.label == '-   Malicious   C&C'), 'label'] = 'C&C'
df.loc[(df.label == '(empty)   Malicious   C&C'), 'label'] = 'C&C'
df.loc[(df.label == '-   Malicious   Attack'), 'label'] = 'Attack'
df.loc[(df.label == '(empty)   Malicious   Attack'), 'label'] = 'Attack'
df.loc[(df.label == '-   Malicious   C&C-HeartBeat'), 'label'] = 'C&C-HeartBeat'
df.loc[(df.label == '(empty)   Malicious   C&C-HeartBeat'), 'label'] = 'C&C-HeartBeat'
df.loc[(df.label == '-   Malicious   C&C-FileDownload'), 'label'] = 'C&C-FileDownload'
df.loc[(df.label == '-   Malicious   C&C-Torii'), 'label'] = 'C&C-Torii'
df.loc[(df.label == '-   Malicious   C&C-HeartBeat-FileDownload'), 'label'] = 'C&C-HeartBeat-FileDownload'
df.loc[(df.label == '-   Malicious   FileDownload'), 'label'] = 'FileDownload'
df.loc[(df.label == '-   Malicious   C&C-Mirai'), 'label'] = 'C&C-Mirai'
df.loc[(df.label == '-   Malicious   Okiru-Attack'), 'label'] = 'Okiru-Attack'


df = df.drop(columns=['ts','uid', 'id.orig_h', 'id.orig_p', 'id.resp_h', 'id.resp_p', 'service', 'local_orig', 'local_resp', 'history'])

df = pd.get_dummies(df, columns=['proto'])
df = pd.get_dummies(df, columns=['conn_state'])

df['duration'] = df['duration'].str.replace('-','0')
df['orig_bytes'] = df['orig_bytes'].str.replace('-','0')
df['resp_bytes'] = df['resp_bytes'].str.replace('-','0')

df.fillna(-1, inplace=True)

print(df.head(5))
print()

df.to_csv("iot23_combined.csv", index=False)