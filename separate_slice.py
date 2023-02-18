import argparse
from glob import glob
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
from skimage.transform import resize

from utils import load_json



SRC_ROOT_PATH = './data/brats'
DST_ROOT_PATH = './data/brats_separated'
MODALITIES = ['t1', 't1ce', 't2', 'flair', 'seg']
IMAGE_MODALITIES = ['t1', 't1ce', 't2', 'flair']
IMAGE_SIZE = (240, 240, 155)


def get_patient_paths(base_dir_path):
    patient_ids = os.listdir(base_dir_path)
    return [os.path.join(base_dir_path, p) for p in patient_ids]


def z_score_normalize(image):
    image = image.astype(np.float32)
    mask = image > 0 
    mean = np.mean(image[mask])
    std = np.std(image[mask])
    image -= mean 
    image /= std 
    return image 


def separate_slices(patient_paths, dst_dir_path, slice_size):
    for patient_path in tqdm(patient_paths):
        patient_id = os.path.basename(patient_path)

        volumes = {}
        for modality in MODALITIES:
            volume_path = os.path.join(patient_path, '{}_{}.nii.gz'.format(patient_id, modality))
            volume = nib.load(volume_path).get_fdata()
            assert volume.shape == IMAGE_SIZE

            if modality in IMAGE_MODALITIES:
                volume = z_score_normalize(volume)
                volume = volume.astype(np.float32)

            else:
                volume = volume.astype(np.int32)

            volumes[modality] = volume

        seg_volume = volumes['seg']
        for slice_num in range(seg_volume.shape[2]):
            seg_slice = seg_volume[..., slice_num]

            if seg_slice.shape != (slice_size, slice_size):
                seg_slice = resize(seg_slice, (slice_size, slice_size))

            class_name = 'Abnormal' if seg_slice.max() > 0 else 'Normal'

            dst_patient_dir_path = os.path.join(dst_dir_path, class_name, patient_id)
            os.makedirs(dst_patient_dir_path, exist_ok=True)

            for img_modality in IMAGE_MODALITIES:
                img_volume = volumes[img_modality]
                img_slice = img_volume[..., slice_num]

                if img_slice.shape != (slice_size, slice_size):
                    img_slice = resize(img_slice, (slice_size, slice_size))

                img_save_path = os.path.join(
                    dst_patient_dir_path, '{}_{}_{}.npy'.format(
                        patient_id, str(slice_num).zfill(4), img_modality
                    )
                )

                np.save(img_save_path, img_slice)

            seg_save_path = os.path.join(
                dst_patient_dir_path, '{}_{}_{}.npy'.format(
                    patient_id, str(slice_num).zfill(4), 'seg'
                )
            )

            np.save(seg_save_path, seg_slice)
        

if __name__ == "__main__":

    train_patient_paths = get_patient_paths(os.path.join(SRC_ROOT_PATH, 'MICCAI_BraTS_2019_STD/MICCAI_BraTS_2019_Data_Training/HGG')) \
                        + get_patient_paths(os.path.join(SRC_ROOT_PATH, 'MICCAI_BraTS_2019_STD/MICCAI_BraTS_2019_Data_Training/LGG'))

    val_test_patient_paths = get_patient_paths(os.path.join(SRC_ROOT_PATH, 'MICCAI_BraTS_2019_STD/MICCAI_BraTS_2019_Data_Validation')) \
                           + get_patient_paths(os.path.join(SRC_ROOT_PATH, 'MICCAI_BraTS_2019_STD/MICCAI_BraTS_2019_Data_Testing'))

    assert len(train_patient_paths) == 335 
    assert len(val_test_patient_paths) == 291 

    parser = argparse.ArgumentParser(description='Unsupervised Lesion Detection')
    parser.add_argument('-c', '--config', help='training config file', required=True)
    args = parser.parse_args()

    config = load_json(args.config)

    separate_slices(train_patient_paths, os.path.join(DST_ROOT_PATH, 'MICCAI_BraTS_2019_Data_Training'), config.dataset.image_size)
    separate_slices(val_test_patient_paths, os.path.join(DST_ROOT_PATH, 'MICCAI_BraTS_2019_Data_Val_Testing'), config.dataset.image_size)