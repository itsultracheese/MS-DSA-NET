
import os
import shutil
import tempfile
import glob
import random
import numpy as np
import ants
import nibabel as nib
import SimpleITK as sitk
mni152_path = './inputs/MNI152_T1_1mm.nii.gz'

def preprocess_fsl(params):
    # robunstfov -i xx -r xx (output) -m xx
    # 1) reorient2std; 2) robust fov; 3) register flair to t1; 4) register all to mni152
    data_dir = os.path.join(params['data_dir'], 'raw')
    save_dir = os.path.join(params['data_dir'], 'fsl')
    os.makedirs(save_dir, exist_ok=True)

    t1_files = sorted( glob.glob(os.path.join(data_dir, "*/t1.nii.gz"), recursive=True) )
    fl_files = sorted( glob.glob(os.path.join(data_dir, "*/flair.nii.gz"), recursive=True) )

    sub_count = len(t1_files)
    for i in range(sub_count):
        name = t1_files[i].split('/')[-2]
        sub_save_dir = os.path.join(save_dir, name)
        os.makedirs(sub_save_dir, exist_ok=True)
            
        sub_t1_path = t1_files[i]
        sub_fl_path = fl_files[i]

        #========================================================
        #origin to std
        sub_t1_std = os.path.join(sub_save_dir, 't1_std.nii.gz')
        mat_ori2std = os.path.join(sub_save_dir, 'ori2std.mat')
        strcmd = 'fslreorient2std -m {} {} {}'.format(mat_ori2std, sub_t1_path, sub_t1_std)
        os.system(strcmd)
        
        #robust fov
        sub_t1_fov = os.path.join(sub_save_dir, 't1_fov.nii.gz')
        mat_fov2std = os.path.join(sub_save_dir, 'fov2std.mat')
        strcmd = 'robustfov -i {} -r {} -m {}'.format(sub_t1_std, sub_t1_fov, mat_fov2std)
        os.system(strcmd)

        sub_t1_reg = os.path.join(sub_save_dir, 't1_reg.nii.gz')
        mat_fov2mni = os.path.join(sub_save_dir, 'fov2mni.mat')
        strcmd = 'flirt -in {} -ref {} -out {} -omat {} -dof 12 -cost corratio  \
                -bins 256 -interp trilinear \
                -searchrx -90 90 -searchry -90 90 -searchrz -90 90'.format(sub_t1_fov, mni152_path, sub_t1_reg, mat_fov2mni)
        os.system(strcmd)

        #convert_xfm -omat ${T1}_nonroi2roi.mat -inverse ${T1}_roi2nonroi.mat
        mat_std2fov = os.path.join(sub_save_dir, 'std2fov.mat')
        strcmd = 'convert_xfm -omat {} -inverse {}'.format(mat_std2fov, mat_fov2std)
        os.system(strcmd)

        mat_ori2fov = os.path.join(sub_save_dir, 'ori2fov.mat')
        strcmd = 'convert_xfm -omat {} -concat {} {}'.format(mat_ori2fov, mat_std2fov, mat_ori2std)
        os.system(strcmd)

        mat_ori2mni = os.path.join(sub_save_dir, 'ori2mni.mat')
        strcmd = 'convert_xfm -omat {} -concat {} {}'.format(mat_ori2mni, mat_fov2mni, mat_ori2fov)
        os.system(strcmd)

        strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp trilinear -applyxfm'.format(sub_t1_path, mni152_path, sub_t1_reg, mat_ori2mni)
        os.system(strcmd)

        #========================================================
        #reg flair to t1-original, then to mni152
        sub_fl_reg0 = os.path.join(sub_save_dir, 'flair_reg0.nii.gz')
        mat_fl2t1 = os.path.join(sub_save_dir, 'mat_fl2t1.mat')
        strcmd = 'flirt -in {} -ref {} -out {} -omat {} -dof 6 -cost mutualinfo  \
                -bins 256 -interp trilinear \
                -searchrx -90 90 -searchry -90 90 -searchrz -90 90'.format(sub_fl_path, sub_t1_path, sub_fl_reg0, mat_fl2t1)
        os.system(strcmd)

        sub_fl_reg = os.path.join(sub_save_dir, 'flair_reg.nii.gz')
        strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp trilinear -applyxfm'.format(sub_fl_reg0, sub_t1_reg, sub_fl_reg, mat_ori2mni)
        os.system(strcmd)

        #========================================================
        gt_files = sorted( glob.glob(os.path.join(data_dir, name, "gt.nii.gz"), recursive=True) )
        if len(gt_files)==0:
            continue

        sub_gt_path = gt_files[0]
        sub_gt_reg0 = os.path.join(sub_save_dir, 'gt_reg0.nii.gz')
        strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp nearestneighbour -applyxfm'.format(sub_gt_path, sub_t1_path, sub_gt_reg0, mat_fl2t1)
        os.system(strcmd)

        sub_gt_reg = os.path.join(sub_save_dir, 'gt_reg.nii.gz')
        strcmd = 'flirt -in {} -ref {} -out {} -init {} -interp nearestneighbour -applyxfm'.format(sub_gt_reg0, sub_t1_reg, sub_gt_reg, mat_ori2mni)
        os.system(strcmd)

    params['data_dir'] = save_dir
    return params
