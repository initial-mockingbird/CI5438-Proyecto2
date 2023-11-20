from src.config.preprocessing import builder
import yaml


def read_data(conf_location):
  with open(conf_location,'r') as f:
    unsafe_config = yaml.safe_load(f)
    # print(unsafe_config)
    path   = unsafe_config["data_path"]
    target =  unsafe_config["target"]
    normalize = unsafe_config["normalize"]
    split = unsafe_config["split"]
    col_dict = unsafe_config["columns"]
    res = builder(path,target,normalize,split,col_dict)
    
    return {"split": split, "target": target, "data": res}