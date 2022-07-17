#provides functions to store and load objects from files 
import pickle

def saveObject(obj, fileName):
    """"Save an object using the pickle library on a file
    
    :param obj: undefined. Object to save
    :param fileName: str. Name of the file of the object to save
    """
    print("Saving " + fileName)
    with open("dumped_objects/" + fileName + ".pkl", 'wb') as fid:
        pickle.dump(obj, fid)
    
def loadObject(fileName):
    """"Load an object from a file
    
    :param fileName: str. Name of the file of the object to load
    :return: obj: undefined. Object loaded
    """
    try:
        with open("dumped_objects/" + fileName + '.pkl', 'rb') as fid:
            obj = pickle.load(fid)
            return obj
    except IOError:
        return None