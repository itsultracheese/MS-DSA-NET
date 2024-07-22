import os
import shutil
import tempfile
import glob
import random
import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" 
device_ids = [0,1]
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np 
import nibabel as nib
import torch

from monai.inferers import sliding_window_inference
from get_model import get_model
from get_transforms import get_test_transforms

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.transforms import (
    SaveImaged,
)
from preprocess_data import preprocess_fsl

#==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = dict()

params['base_dir'] = './outputs'
params['model_name'] = './pretrained/model.pth'
params['data_dir'] = './inputs/'
params['model_type'] = 'MS_DSA_NET' 
params['sa_type'] = 'parallel'
params['chans_in']   = 2
params['chans_out']  = 2
params['feature_size'] = 16 
params['project_size'] = 64 #dsa projection size
params['patch_size'] = [128]*3 
params['num_workers'] = 2
params['seq'] = 't1+t2'
params['chans_in'] = 2
params['date'] = datetime.date.today()

os.makedirs(params['base_dir'], exist_ok=True)

def test(model, params):

    pretrain = params['model_name']
    if os.path.exists(pretrain):
        model.load_state_dict( torch.load(pretrain) )
        print('pretrained model ' + pretrain + ' loaded')
    else:
        print('no pretrained model found')

    model.to(device=device)
    model.eval()

    test_transform, post_transform = get_test_transforms(params)
    test_dict = get_data(params)
    test_ds = CacheDataset(data=test_dict, transform=test_transform, cache_num=4, cache_rate=1, num_workers=params['num_workers'])
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=params['num_workers'], pin_memory=True)

    epoch_iterator_val = tqdm(test_loader, desc="test (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    
    metrics = dict()
    with torch.no_grad():
        idx = 0
        for batch in epoch_iterator_val:
            idx += 1

            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name.split('.')[0]

            val_inputs = batch["image"].to(device)
            val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 2, model, overlap = 0.25)
            batch['pred'] = val_outputs
 
            output_dir = os.path.join(params['save_dir'], img_name)
            os.makedirs(output_dir, exist_ok=True)

            batch = [post_transform(i) for i in decollate_batch(batch)]
            #val_preds = val_inputs['pred'].cpu().numpy().astype(np.uint8)
            output_name = os.path.join(output_dir, 'seg.nii.gz')
            SaveImaged(keys="pred", output_dir=output_dir, output_postfix="seg", resample=False, separate_folder=False)(batch[0])


def get_data(params):
    t1_files = sorted( glob.glob(os.path.join(params["data_dir"], "./*/t1_reg.nii.gz"),  recursive=True) )
    fl_files = sorted( glob.glob(os.path.join(params["data_dir"], "./*/flair_reg.nii.gz"), recursive=True) )

    data_dict = []
    for t1_f, fl_f in zip(t1_files, fl_files):
        if params['seq'] == 't1':
            data_dict.append({'image':t1_f})
        elif params['seq'] == 't2':
            data_dict.append({'image':fl_f})
        else:
            data_dict.append({'image':[t1_f, fl_f]})
    
    return data_dict

def main(params):
    
    params = preprocess_fsl(params)

    model, params = get_model(params)

    params['save_dir'] = os.path.join(params['base_dir'] , str(params['date']))
    os.makedirs(params['save_dir'], exist_ok=True)
    
    test(model, params)
    

if __name__ == '__main__':
    main(params=params)
