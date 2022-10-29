import pickle
import pandas as pd

file = open('./data/hf/hf/hf_code2idx_new.pickle','rb')

data = pickle.load(file)
file.close()

# df = pd.DataFrame(data)
i = 0

for elem in data:
    print(elem)
    i+=1
    if i >=5:
        break