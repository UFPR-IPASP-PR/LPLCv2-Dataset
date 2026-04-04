from natsort import natsorted
from glob import glob
import json
from pathlib import Path

def dict_to_string(dct):
    # For logging purposes
    ret = ""
    for k, v in dct.items():
        ret += f"| {k}: {v}"
    return ret

def dict_to_table(dct):
    keys = ['epoch', 'train_acc', 'val_acc', 'test_acc']
    ret = ""
    #for k in dct.keys():
    for k in keys:
        ret += f" {k:<15}|"
    ret += "\n"
    #ret += f"{'-'*(16*len(dct.keys()) + len(dct.keys()))}"
    ret += f"{'-'*(16*len(keys) + len(keys))}"
    ret += "\n"
    #for v in dct.values():
    #    ret += f" {v:<15.4g}|"
    for k in keys:
        ret += f" {dct[k]:<15.4g}|"
    return ret

def find_model(save_path):
    # By the syntax the outputted models, the last one in the list is the latest
    models = natsorted(list(glob(f"{save_path}/*.pth")))
    for m in models:
        if "best" in m:
            print("Found best model ", m)
            return m
    chosen = m
    print("Found last model ", m)
    return chosen

def start_log(log_file="logs/log.log"):
    with open(log_file, "w") as fd:
        fd.write("[")

def log_metrics_json(metrics, log_file="logs/log.log"):
    with open(log_file, "a") as fd:
        fd.write(json.dumps(metrics) + ",\n")

def end_log(log_file="logs/log.log"):
    with open(log_file, "a") as fd:
        fd.write("\{\}]")

def log_config(save_path, cfg):
    cfg['dataset']['path'] = str(cfg['dataset']['path'])
    with open(save_path / Path("experiment_config.json"), "w") as fd:
        json.dump(cfg, fd, indent=2)
