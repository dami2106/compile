import numpy as np 

def get_seq(segments = 4, symbols = 4, length = 8):
    seq = []
    symbols = np.random.choice(
        np.arange(1, symbols + 1), segments, replace=False)
    for seg_id in range(segments):
        segment_len = np.random.choice(np.arange(1, 5))
        seq += [symbols[seg_id]] * segment_len
    
    while len(seq) < length - 1:
        seq.append(seq[len(seq) -  1])  
    
    seq = seq[:length - 1]
    # seq += [0]

    return seq

def save_sequence(nb_traces = 5000, segments = 4, symbols = 4, length = 10):
    data_states = np.zeros([nb_traces, length - 1, 4], dtype='float32')
    data_actions = np.zeros([nb_traces, length - 1], dtype='long')
        
    for i in range(nb_traces):
        seq = get_seq(segments, symbols, length)
        len_c = 0 
        for elem in seq:
            # data_states[i][len_c] = np.random.rand(1, 4)
            data_states[i][len_c] = (np.ones((1, 4)) * elem)
            data_actions[i][len_c] = elem 
            len_c += 1
    
    np.save('trajectories/sequence/states', data_states)
    np.save('trajectories/sequence/actions', data_actions)


if __name__ == '__main__':
    save_sequence(5000, 4, 4, 10)