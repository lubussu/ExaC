import pickle


def saveObject(obj, fileName):
    with open(fileName + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)


def loadObject(fileName):
    try:
        with open(fileName + '.pkl', 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None
