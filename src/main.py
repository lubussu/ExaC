import integration as di
import build_model2 as bm

di.data_integration()
bm.build_model("../dataset/final_dataset/k2-kepler.csv")