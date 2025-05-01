import os
import json
import shutil


def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)


def maybe_mkdir_p(directory):
    directory = os.path.abspath(directory)
    splits = directory.split("/")[1:]
    for i in range(0, len(splits)):
        if not os.path.isdir(os.path.join("/", *splits[:i + 1])):
            try:
                os.mkdir(os.path.join("/", *splits[:i + 1]))
            except FileExistsError:
                # this can sometimes happen when two jobs try to create the same directory at the same time,
                # especially on network drives.
                print("WARNING: Folder %s already existed and does not need to be created" % directory)


def subdirs(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [l(folder, i) for i in os.listdir(folder) if os.path.isdir(os.path.join(folder, i))
           and (prefix is None or i.startswith(prefix))
           and (suffix is None or i.endswith(suffix))]
    if sort:
        res.sort()
    return res


base = "/data/DATASET3/nnUNet_raw/nnUNet_raw_data/Task10_Colon"
out = "/data/DATASET3/nnUNet_raw/nnUNet_raw_data/Task010_Colon"
cases = ['imagesTr', 'labelsTr']
# cases = subdirs(base, join=False) # ['imagesTr', 'labelsTr']

maybe_mkdir_p(out)
maybe_mkdir_p(os.path.join(out, "imagesTr"))
maybe_mkdir_p(os.path.join(out, "imagesTs"))
maybe_mkdir_p(os.path.join(out, "labelsTr"))
maybe_mkdir_p(os.path.join(out, "labelsTs"))

tr = os.listdir(os.path.join(base, cases[0])) # imagesTr pancreas_001.nii.gz pancreas_002.nii.gz ……
case_id = 0
dataLen = len(tr)
for t in tr:  # pancreas_001.nii.gz
    # case_id = int(t.split("_")[-1][:3]) # '377'
    case_id = case_id + 1
    if case_id <= int(dataLen*0.9):
        shutil.copy(os.path.join(base, cases[0], t), os.path.join(out, "imagesTr", t.split(".")[0] + "_0000.nii.gz"))
        shutil.copy(os.path.join(base, cases[1], t), os.path.join(out, "labelsTr", t))
    elif case_id <= dataLen:
        shutil.copy(os.path.join(base, cases[0], t), os.path.join(out, "imagesTs", t.split(".")[0] + "_0000.nii.gz"))
        shutil.copy(os.path.join(base, cases[1], t), os.path.join(out, "labelsTs", t))
    print(case_id, " ", t, ' done!')

json_dict = {}
json_dict['name'] = "Colon"
json_dict['description'] = "colon cancer primaries"
json_dict['tensorImageSize'] = "4D"
json_dict['reference'] = "Memorial Sloan Kettering Cancer Center "
json_dict['licence'] = "CC-BY-SA 4.0"
json_dict['release'] = "1.0 06/05/2018"
json_dict['modality'] = {
    "0": "CT",
}
json_dict['labels'] = {
    "0": "background",
    "1": "colon cancer primaries"
}
json_dict['numTraining'] = int(dataLen*0.9)
json_dict['numTest'] = dataLen - int(dataLen*0.9)
json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in
                         tr[:int(dataLen*0.9)]] # dataset里面不需要"_000",但是文件里面需要
json_dict['test'] = ["./imagesTs/%s" % i for i in
                     tr[int(dataLen*0.9):]]

save_json(json_dict, os.path.join(out, "dataset.json"))
