import numpy as np
def fetch_batch(data, date, batch_size, l, w, pred_window):
    num_steps = l*w
    data_len, features = data.shape
    end = data_len-num_steps-pred_window
    index = np.random.randint(0,end,batch_size)
    X = []
    y = []
    y_date = []
    for i in index:
        temp_X = data[i:i+num_steps,:].reshape(l,w,features)
        temp_y = data[i+num_steps:i+num_steps+pred_window,0].reshape(-1)
        # date only corrosponds to the labels that is y
        temp_date = date[i+num_steps:i+num_steps+pred_window]
        X.append(temp_X)
        y.append(temp_y)
        y_date.append(temp_date)
    return np.array(X), np.array(y), np.array(y_date)