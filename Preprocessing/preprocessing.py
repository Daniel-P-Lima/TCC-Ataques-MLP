import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
GENERATE_STD_DATA = True

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

# Remove C&C-Mirai due to low amount of instances
df = df[df["label"] != "C&C-Mirai"]

if GENERATE_STD_DATA:
    X = df
    y = X["label"]

    X = X.drop("label", axis=1)

    print(y.value_counts())

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y)

    scaler = StandardScaler()

    Xtrain = scaler.fit_transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    pd.DataFrame(Xtrain).to_csv("../Data/Xtrain.csv", index=False)
    pd.DataFrame(Xtest).to_csv("../Data/Xtest.csv", index=False)
    pd.DataFrame(ytrain).to_csv("../Data/ytrain.csv", index=False)
    pd.DataFrame(ytest).to_csv("../Data/ytest.csv", index=False)
else:
    df.to_csv("../Data/iot23_combined.csv", index=False)