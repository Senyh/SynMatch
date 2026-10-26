import os
import h5py
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter

slice_num = 0
file_list = []
scribble_path = 'YOUR_DATA_PATH'
scribble_files = sorted([os.path.join(scribble_path, x, y)
                         for x in os.listdir(scribble_path) if x.endswith('scribbles')
                         for y in os.listdir(os.path.join(scribble_path, x)) if y.endswith('scribble.nii.gz')])
for i in tqdm(range(len(scribble_files))):
    scribble_file = scribble_files[i]
    scribble_itk = sitk.ReadImage(scribble_file)
    origin = scribble_itk.GetOrigin()
    spacing = scribble_itk.GetSpacing()
    direction = scribble_itk.GetDirection()
    scribble = sitk.GetArrayFromImage(scribble_itk)
    
    case_name = scribble_file.split('/')[-1].split('.')[0][:-9]
    if case_name[:-2] != "patient090_frame":
        case_name = case_name[:-2] + '02' if not case_name.endswith('01') else case_name
        h5f = h5py.File(scribble_path + "/data/{}.h5".format(case_name), "r")
        image = h5f["image"][:]
        for slice_idx in range(image.shape[0]):
            f = h5py.File(scribble_path + '/data/slices_image_scribble/{}_slice_{}.h5'.format(case_name, slice_idx+1), 'w')
            f.create_dataset('image', data=image[slice_idx], compression="gzip")
            f.create_dataset('label', data=scribble[slice_idx], compression="gzip")
            f.close()
            slice_num += 1
            file_list.append('{}_slice_{}.h5'.format(case_name, slice_idx+1))
    else:
        case_name = case_name[:-2] + '01' if case_name.endswith('04') else case_name[:-2] + '02'
        h5f = h5py.File(scribble_path + "/data/{}.h5".format(case_name), "r")
        image = h5f["image"][:]
        for slice_idx in range(image.shape[0]):
            f = h5py.File(scribble_path + '/data/slices_image_scribble/{}_slice_{}.h5'.format(case_name, slice_idx+1), 'w')
            f.create_dataset('image', data=image[slice_idx], compression="gzip")
            f.create_dataset('label', data=scribble[slice_idx], compression="gzip")
            f.close()
            slice_num += 1
            file_list.append('{}_slice_{}.h5'.format(case_name, slice_idx+1))
print("Converted all ACDC volumes to 2D slices with scribbles")
print("Total {} slices".format(slice_num))

counts = Counter(file_list)

duplicates = [item for item, count in counts.items() if count > 1]
print(duplicates)