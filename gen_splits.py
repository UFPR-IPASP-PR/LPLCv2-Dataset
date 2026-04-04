import json
import random
from glob import glob
from pathlib import Path
import shutil
import argparse
import os
import copy
 

def save_folds(tr, vl, te, idx=0, dataset_dir="lpr_dts", fold_dir="folds"):
    path = Path(f"{fold_dir}")
    path.mkdir(parents=True, exist_ok=True)    

    out = {
        "train": tr,
        "val": vl,
        "test": te
    }
    with open(f"{fold_dir}/fold_{idx}.json", "w") as fd:
        json.dump(out, fd, indent=2)


def split_files(files, n):
    ret = []
    for i in range(n):
        ret.append([])
    i = 0
    for f in files:
        ret[i%n].append(f)
        i += 1
    return ret

def agg_at_idxs(pls, idxs, cls):
    ret = {}
    for i in cls:
        ret[i] = []
        for idx in idxs:
            ret[i] += pls[int(i)][idx]
    return ret

def join_classes(pls, cls):
    ret = []
    for i in cls:
        ret += pls[int(i)]
    return ret

def gen_sym_partition(dataset_dir, fs, sldir, ann, fname, c_cfg):
    path = Path(sldir) / Path(fname)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

    n_classes = len(c_cfg['all_classes'])
    c_dct = c_cfg['class_dct']
    c_dct = {x: int(k) for k,v in c_dct.items() for x in v}
    for i in range(n_classes):
        c_path = path / Path(f"{i}")
        c_path.mkdir(parents=True, exist_ok=True)

    for im in fs:
        bn = os.path.basename(im)
        split = os.path.basename(bn).split("_")
        k = split[0]
        idx = split[1]

        leg = ann[k]['anns'][int(idx)]['leg']
        if leg not in c_dct.keys():
            continue
        leg = c_dct[leg]

        c_path = path / Path(f"{leg}")
        src = Path(Path(dataset_dir) / Path(bn)).resolve()
        if not os.path.exists(src):
            continue
        os.symlink(f"{src}", Path(c_path) / Path(os.path.basename(im)))

def gen_sym_links(dataset_dir, tr, vl, te, fold_name, ann_file, c_cfg, sldir):
    gen_sym_partition(dataset_dir, tr, sldir, ann_file, fold_name + "/train/", c_cfg)
    gen_sym_partition(dataset_dir, vl, sldir, ann_file, fold_name + "/val/", c_cfg)
    gen_sym_partition(dataset_dir, te, sldir, ann_file, fold_name + "/test/", c_cfg)

def gen_splits(cfg, ann_file):
    js = ann_file
    nf = cfg['n_folds']
    classes = [x for x in range(cfg['n_classes'])]
    lp_names = [{'fname': k, 'idx': idx, 'leg': x['leg'], 'ocr': x['ocr']}
                for k, v in js.items() for idx,x in enumerate(v['anns'])]
    plates = {i: [f"{x['fname']}_{x['idx']}_{x['ocr']}.jpg"
                for x in lp_names if x['leg'] == i] for i in classes}

    for i in classes:
        plates[i] = split_files(plates[i], nf)

    n_folds = {}
    for i in range(nf):
        valid_idx = i
        valid_fs = join_classes(agg_at_idxs(plates, [valid_idx], cls=classes), classes)

        test_idxs = []
        for j in range(nf//2):
            test_idxs.append((valid_idx + j + 1) % nf)
        test_fs = join_classes(agg_at_idxs(plates, test_idxs, cls=classes), classes)

        train_idxs = []
        for j in range(nf//2):
            train_idxs.append((valid_idx - j - 1) % nf)
        train_fs = join_classes(agg_at_idxs(plates, train_idxs, cls=classes), classes)

        save_folds(train_fs, valid_fs, test_fs, f"{i}_1",
                   fold_dir=cfg['output_dir'])
        n_folds[f"{i}_1"] = {"train": copy.deepcopy(train_fs),
                             "val": copy.deepcopy(valid_fs),
                             "test": copy.deepcopy(test_fs)}

        if cfg['cross_fold']:
            save_folds(test_fs, valid_fs, train_fs, f"{i}_2",
                    fold_dir=cfg['output_dir'])
            n_folds[f"{i}_2"] = {"train": copy.deepcopy(test_fs),
                                 "val": copy.deepcopy(valid_fs),
                                 "test": copy.deepcopy(train_fs)}

    return n_folds

def load_splits(cfg, c_cfg):
    fdir = Path(cfg['output_dir'])# / Path(cfg['sub_dir'])

    print(fdir)
    all_folds = sorted(glob(f"{fdir}/*.json"))
    ret = {}
    for f in all_folds:
        k = f.split("_")
        k = k[-2] + "_" + k[-1].split(".")[0]
        with open(f, "r") as fd:
            ret[k] = json.load(fd)
    return ret

def gen_sldirs(folds, cfg, ann_file, subdir=""):
    for k,v in folds.items():
        gen_sym_links(cfg['dataset_dir'], v['train'], v['val'], v['test'],
                      k, ann_file, cfg['class_config'],
                      sldir=os.path.join(cfg['sym_link_dir'], cfg["sub_dir"])
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--annotation_file', default="working_dataset.json")
    parser.add_argument('--output_dir', default="./LPLCv2/folds")
    parser.add_argument('--do_shuffle', default=False, action='store_true')
    parser.add_argument('--n_folds', default=5)
    parser.add_argument('--cross_fold', default=False, action='store_true')
    parser.add_argument('--gen_sym_links', default=False, action='store_true')
    parser.add_argument('--sym_link_dir', default="sldir")

    parser.add_argument('--load_folds', action='store_true')

    args = vars(parser.parse_args())
    cfg = {}

    if args['config'] is None:
        cfg = copy.deepcopy(args)
        ann_file = args['annotation_file']
    else:
        with open(args['config'], "r") as fd:
            cfg = json.load(fd)
        cfg['load_folds'] = args['load_folds']
        ann_file = cfg['annotation_file']
    cfg['n_classes'] = 4
    with open(ann_file, "r") as fd:
        ann = json.load(fd)

    if not cfg['load_folds']:
        folds = gen_splits(cfg, ann)
    else:
        folds = load_splits(cfg, cfg['class_config'])

    if cfg['gen_sym_links']:
        gen_sldirs(folds, cfg, ann)

