import numpy as np
import itertools
# read data from file name

def read_data(filename):
    data= open(filename, 'r').read()
    data=data.lower()
    chars=list(set(data))
    data_size, vocab_size = len(data), len(chars)
    print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
    return chars,data

# read dinoudar  names line by line
def read_dinousar_names(filename):
    with open(filename) as f:
        dinousar_names = f.readlines()
        dinousar_names = [x.lower().strip() for x in dinousar_names ]
    return dinousar_names


# create X and Y training set
def create_x_y_dataset(dinousar_names,char_to_ix):
    X=[]
    Y=[]
    for name in dinousar_names:
        x=[0] + [char_to_ix[ch] for ch in name]
        y= x[1:] + [char_to_ix['\n']]
        X.append(x)
        Y.append(y)
        
    #pad with zeros for maximum length
    X=np.array(list(itertools.zip_longest(*X, fillvalue=0))).T
    Y=np.array(list(itertools.zip_longest(*Y, fillvalue=0))).T
    X=X.reshape(-1,1,27)
   
    
    
    return X,Y