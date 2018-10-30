import numpy as np
def fetch_batch(data,batch_size,n_steps):
    data_len = len(data)
    start = np.random.randint(0,data_len-n_steps,batch_size)
    prep_data = np.array([data[i:i+n_steps+1,:] for i in start])
    return (prep_data[:,:-1,:], prep_data[:,1:,0:1])