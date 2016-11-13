import pickle
def change_pickle_protocol(src, target, protocol=2):
    with open(src,'rb') as f:
        obj = pickle.load(f)
    with open(target,'wb') as f:
        pickle.dump(obj,f,protocol=protocol)

