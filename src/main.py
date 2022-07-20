import integration as di
import build_model as bm
import classifier as cf

di.data_integration()

bm.build_model("../dataset/final_dataset/k2-kepler_lc.csv")
cf.classify(0)

bm.build_model("../dataset/final_dataset/k2-kepler.csv")
cf.classify(1)
