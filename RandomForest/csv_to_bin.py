import struct

FILE_PATHNAME = "../Data/iot23_single_instance" # Without extension

def strToBool(str: str):
    return str == "True"

def splitListFormat(list: list):
    types1 = [float] + [float for _ in range(7)] + [strToBool for _ in range(16)]
    types2 = [float] + [int for _ in range(7)] + [int for _ in range(16)]
    
    for i in range(len(types1)):
        list[i] = types1[i](list[i])
        list[i] = types2[i](list[i])

if __name__ == "__main__":
    with open(f"{FILE_PATHNAME}.csv", "r") as input_file:
        with open(f"{FILE_PATHNAME}.bin", "wb") as output_file:
            input_file.readline() # consume header

            for line in input_file:
                line = line.strip()
                features = line.split(',')
                del features[8]
                splitListFormat(features)
                output_file.write(struct.pack(f"<f{'I' * 7}{'?' * 16}", *features))
                # print(features)