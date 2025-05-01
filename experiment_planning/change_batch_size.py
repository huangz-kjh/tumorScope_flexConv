from batchgenerators.utilities.file_and_folder_operations import *
import numpy as np

if __name__ == '__main__':
    input_file = '/data/DATASET3/nnUNet_preprocessed/Task007_Pancreas/nnUNetPlansv2.1_plans_3D.pkl'
    output_file = '/data/DATASET3/nnUNet_preprocessed/Task007_Pancreas/nnUNetPlansv2.12_plans_3D.pkl'
    a = load_pickle(input_file)
    # a['plans_per_stage'][0]['batch_size'] = 2
    # a['plans_per_stage'][1]['batch_size'] = 2
    a['plans_per_stage'][0]['patch_size'] = np.array([64,192,192])
    a['plans_per_stage'][1]['patch_size'] = np.array([64,192,192])
    save_pickle(a, output_file)