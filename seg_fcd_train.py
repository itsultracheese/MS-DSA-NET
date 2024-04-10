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
from monai.config import print_config
from monai.metrics import DiceMetric, compute_average_surface_distance
from utils2 import write_image, post_process_segment, EarlyStopping
from utils2 import evaluate_classifcation, seed_torch, evaluate_fp, evaluate_dice

from get_datasets import FcdDataset
from get_model import get_model
from get_transforms import get_fcd2_transforms
from get_loss import get_loss, get_lrschedule

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
)
from monai.transforms import AsDiscrete

#==================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = dict()
params['model_type'] = 'MS_DSA_NET' 
params['sa_type'] = 'parallel' #parallel, serial, spatial, channel, none
params['chans_in']   = 2
params['chans_out']  = 2
params['feature_size'] = 16 #16,32; 24 for swinunetr; 32 for uneter
params['project_size'] = 64 #dsa projection size
params['batch_size'] = 1
params['samples_per_case'] = 4 #2,4,6;
params['patch_size'] = [128]*3 
params['learning_rate'] = 1e-4 
params['lrschedule'] = 'plateau'
params['loss_type']  = 'DiceCE'  # 'Dice' 'DiceCE' 'Recall' 'gDiceCE'
params['include_background']= False

params['epochs']        = 1000
params['eval_epochs']   = 20
params['warmup_epochs'] = 10
params['num_workers']   = 2

params['use_symmetry'] = False
params['seq'] = 't1+t2' #'t1+t2' #t1, t2, t1+t2
if params['seq'] == 't1' or params['seq'] == 't2':
    params['chans_in'] = 2 if params['use_symmetry'] else 1
else:
    params['chans_in'] = 4 if params['use_symmetry'] else 2

params['base_dir'] = '/home/zhangxd/Projects/Project/MONAI/models/brain/fcd2'
params['model_name'] = 'model.pth'

params['test_mode'] = 1 #0->train; 1->test; 2->evaluate
params['date'] = datetime.date.today()
if params['test_mode']:
    params['date'] = '2024-03-17' #for DualSAUNetr_v1 '2023-12-25'#

params['seed'] = 12345 #3407,12345
seed_torch(params['seed'])
os.makedirs(params['base_dir'], exist_ok=True)

metric_func = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)
post_label = AsDiscrete(to_onehot = params['chans_out'])
post_pred = AsDiscrete(argmax=True, to_onehot=params['chans_out'])

def test(model, test_loader, params):

    output_dir = os.path.join(params['save_dir'], 'test')
    os.makedirs(output_dir, exist_ok=True)
    pretrain = os.path.join(params['save_dir'], params['model_name'] )
    if os.path.exists(pretrain):
        model.load_state_dict( torch.load(pretrain) )
        print('pretrained model ' + pretrain + ' loaded')
    else:
        print('no pretrained model found')

    model.to(device=device)
    model.eval()

    epoch_iterator_val = tqdm(test_loader, desc="test (X / X Steps) (dice=X.X)", dynamic_ncols=True)
    
    metrics = dict()
    with torch.no_grad():
        idx = 0
        for batch in epoch_iterator_val:
            idx += 1

            original_affine = batch["image_meta_dict"]["affine"][0].numpy()
            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-2]
            img_name = img_name.split('.')[0]

            val_labels = batch["label"]
            val_labels = val_labels.cpu().numpy()
            val_labels_onehot = post_label(val_labels[0,...])[np.newaxis,...]

            val_inputs = batch["image"].to(device)
            val_outputs = sliding_window_inference(val_inputs, params['patch_size'], 2, model, overlap = 0.25)
            val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
            val_preds = np.argmax(val_outputs, axis=1).astype(np.uint8)

            metric_lab = dict()
            for l in range(1, params['chans_out']):
                
                metrics_seg = dict()
                lab_prob = val_outputs[0,l,...]
                lab_mask = (val_preds[0] == l).astype(np.uint8)
                lab_mask_post, lab_cc = post_process_segment(lab_mask, 1)

                metrics_seg = evaluate_classifcation(
                    lab_mask_post, 
                    lab_prob, 
                    val_labels_onehot[0, l, :, :, :], 
                    os.path.join(output_dir, img_name)
                )

                fp = evaluate_fp(lab_cc, val_labels_onehot[0, l, :, :, :])
                metrics_seg['fp'] = fp

                #asd = compute_average_surface_distance(lab_mask_post[np.newaxis,np.newaxis,...], val_labels_onehot[:1,l:l+1,...])
                #metrics_seg['asd'] = asd.item()*1.0

                metric_lab[l] = metrics_seg
                metric_keys = metrics_seg.keys()
            
            metrics[img_name] = metric_lab
            lab_keys = metric_lab.keys()

            output_name = os.path.join(output_dir, img_name+'_seg.nii.gz')
            nib.save(nib.Nifti1Image(val_preds[0,...].astype(np.uint8), original_affine), output_name)
            output_name = os.path.join(output_dir, img_name+'_gt.nii.gz')
            nib.save(nib.Nifti1Image(val_labels[0,0,...].astype(np.uint8), original_affine), output_name)

            for ch in range(params['chans_in']):
                val_input = val_inputs.cpu().numpy()[0, ch, :, :, :]
                output_name = os.path.join(output_dir, img_name+'_src{}.nii.gz'.format(ch))
                nib.save(nib.Nifti1Image(val_input.astype(np.float64), original_affine), output_name)
    
    stat_lab = dict()
    stat_det = dict() 
    case_keys = metrics.keys()

    for l in lab_keys: #each label
        stat_lab[l] = dict()
        stat_det[l] = 0

        for m in metric_keys: #each metric
            vals = []
            for k in case_keys: #each case
                val = metrics[k][l][m]
                vals.append( val )
                if m == 'f1' and val > 0:
                    stat_det[l] += 1 

            stat_lab[l][m] = dict()
            stat_lab[l][m]['mean'] = np.mean(vals)
            stat_lab[l][m]['std'] = np.std(vals)
            

    metric_file = os.path.join(output_dir, 'metric_sub.txt')
    with open(metric_file, "w") as file:
        for l in lab_keys:
            file.write("Label:{}======================================\n".format(l))
            file.write("{:>15s}".format('case'))
            for m in metric_keys:
                file.write("{:>15s}".format(m))
            file.write("\n")
            
            for k in case_keys:
                file.write("{:>15s}".format(k))
                for m in metric_keys:
                    file.write("{:15.3f}".format(metrics[k][l][m]))
                file.write("\n")
        
            file.write("{:>15s}".format('mean'))
            for m in metric_keys:
                file.write("{:15.3f}".format(stat_lab[l][m]['mean']) )
            file.write("\n")
            file.write("{:>15s}".format('std'))
            for m in metric_keys:
                file.write("{:15.3f}".format(stat_lab[l][m]['std']) )
            file.write("\n")  

            file.write("{:>15s}".format('sens_sub'))
            file.write("{:15.3f}".format( stat_det[l]/len(metrics) ) )
            file.write("\n")  

def validation(epoch, val_loader, model, params):
    epoch_iterator = tqdm(enumerate(val_loader), total=len(val_loader))

    model.eval()
    with torch.no_grad():
        for step, batch in epoch_iterator:
            val_inputs, val_labels = (batch["image"], batch["label"])
            val_inputs = val_inputs.to(device)
            #val_labels = val_labels.to(device)

            val_outputs = sliding_window_inference(
                inputs = val_inputs,
                roi_size = params['patch_size'], 
                sw_batch_size = 1, 
                predictor = model,
                overlap=0.25, 
                device=torch.device('cpu')
            )

            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            metric_func(y_pred=val_output_convert, y=val_labels_convert)

            epoch_iterator.set_postfix(
            {
                "dice": metric_func.aggregate().item(),
            }
        )
        mean_dice_val = metric_func.aggregate().item()
        metric_func.reset()
    return mean_dice_val

def train(epoch, train_loader, model, optimizer, loss_func, params):
    
    model.train()
    now_lr = optimizer.state_dict()['param_groups'][0]['lr']
    epoch_loss = {'loss':0}
    epoch_iterator = tqdm(enumerate(train_loader), total=len(train_loader))
    epoch_iterator.set_description(f"Epoch {epoch}")
    for step, batch in epoch_iterator:
 
        x, y = (batch["image"].to(device), batch["label"].to(device))
        logit_map = model(x)

        #write_image(x.cpu().numpy()[0,0,...], params['save_dir'], 'patch.nii.gz')

        loss = loss_func(logit_map, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss["loss"]= epoch_loss["loss"] + loss.item()
        epoch_iterator.set_postfix(
            {
                "loss":'{:.3f}'.format( epoch_loss["loss"] / (step + 1) ) ,
                "lr":'{:.8f}'.format( now_lr ) ,
            }
        )

        '''
        p = torch.softmax(logit_map, dim=1).detach()
        write_image(x.cpu().numpy()[0,0,...], save_dir, 'patch_img0.nii.gz')
        write_image(x.cpu().numpy()[0,1,...], save_dir, 'patch_img1.nii.gz')
        write_image(y.cpu().numpy()[0,0,...], save_dir, 'patch_lab.nii.gz')
        write_image(p.cpu().numpy()[0,0,...], save_dir, 'patch_pred.nii.gz')
        '''

def get_data_splits(params):
    data_dir = '/home/zhangxd/Projects/Data/brain/FCD2-public/prep-fsl/fcd'
    t1_files = sorted( glob.glob(os.path.join(data_dir, "*/t1_reg.nii.gz"),  recursive=True) )
    fl_files = sorted( glob.glob(os.path.join(data_dir, "*/flair_reg.nii.gz"), recursive=True) )
    gt_files = sorted( glob.glob(os.path.join(data_dir, "*/roi_reg.nii.gz"), recursive=True) )
    data_dict = []
    for t1_f, fl_f, gt_f in zip(t1_files, fl_files, gt_files):
        if params['seq'] == 't1':
            data_dict.append({'image':t1_f, 'label': gt_f})
        elif params['seq'] == 't2':
            data_dict.append({'image':fl_f, 'label': gt_f})
        else:
            data_dict.append({'image':[t1_f, fl_f], 'label': gt_f})

    np.random.seed(12345)
    np.random.shuffle(data_dict)
    np.random.seed(params['seed'])

    nTrain = int( len(t1_files)*0.8 )
    nTest  = len(t1_files) - nTrain

    nValid = int(nTrain*0.1)
    nTrain = nTrain - nValid
    if params['test_mode']:
        return data_dict[-nTest:]
    else:
        return data_dict[:nTrain], data_dict[nTrain:nTrain+nValid]

def main(params):

    model, params = get_model(params)
    if 'seq' in params.keys() and (params['seq'] == 't1' or params['seq'] == 't2'):
        params['save_dir'] = params['save_dir'] + '_' + params['seq']
    if params['use_symmetry']:
        params['save_dir'] = params['save_dir'] + '_Symm'
    if "DualSAUNetr" in params['model_type']:
        params['save_dir'] = params['save_dir'] + '_' + params['sa_type']

    params['save_dir'] = params['save_dir']  + '_' + str(params['date'])
    os.makedirs(params['save_dir'], exist_ok=True)
    
    params_count = sum(param.numel() for param in model.parameters())
    params['parameters'] = params_count
    print(params)

    train_transforms, valid_transforms, _ = get_fcd2_transforms(params)
    loss_func = get_loss(params)
    
    if params['test_mode'] == 1:
        test_dict = get_data_splits(params)

        #test_ds = CacheDataset(data=valid_dict, transform=valid_transforms, cache_num=4, cache_rate=1, num_workers=params['num_workers'])
        test_ds = FcdDataset(test_dict, params, training=False)
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=params['num_workers'], pin_memory=True)
        test(model, test_loader, params)

    else:
        train_dict, valid_dict = get_data_splits(params)

        metric_file = os.path.join(params['save_dir'], 'valid_metric.txt')
        with open(metric_file, "w") as file:
            for key in params.keys():
                file.write("{}:{}\n".format(key, params[key]))

        data_file = os.path.join(params['save_dir'], 'datalist_train.txt')
        with open(data_file, "w") as file:
            for it in train_dict:
                file.write("{} {} {}\n".format(it["image"][0], it["image"][1],it["label"]))

        data_file = os.path.join(params['save_dir'], 'datalist_valid.txt')
        with open(data_file, "w") as file:
            for it in valid_dict:
                file.write("{} {} {}\n".format(it["image"][0], it["image"][1],it["label"]))

        train_ds = CacheDataset(data=train_dict, transform=train_transforms, cache_num=68, cache_rate=0.5, num_workers=params['num_workers'])
        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, num_workers=params['num_workers'], pin_memory=True)

        valid_ds = CacheDataset(data=valid_dict, transform=valid_transforms, cache_num=5, num_workers=params['num_workers'])
        valid_loader = DataLoader(valid_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        model.to(device=device)
        model = torch.nn.DataParallel(model, device_ids= device_ids) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], weight_decay=1e-5)
        #scheduler = get_lrschedule(optimizer, params)

        dice_val_best = 0.0
        early_stopping = EarlyStopping()
        for epoch in range(params['epochs']):

            train(epoch, train_loader, model, optimizer, loss_func, params)
            #scheduler.step()

            if ((epoch+1) % params['eval_epochs']== 0) or epoch == params['epochs']-1:  
                dice_val = validation(epoch, valid_loader, model, params)
                #scheduler.step(dice_val)

                '''
                early_stopping(dice_val)
                if early_stopping.early_stop:
                    print(
                        "Early stopped at epoch {}".format(epoch+1)
                    )
                    break
                '''

                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    torch.save(model.module.state_dict(), os.path.join(params['save_dir'], params['model_name'] ))
                    print(
                        "Model Was Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
                    
                    metric_file = os.path.join(params['save_dir'], 'valid_metric.txt')
                    with open(metric_file, "a") as file:
                        file.write("epoch {:03d}: {:.3f}\n".format(epoch+1, dice_val_best))

                else:
                    print(
                        "Model Was Not Saved ! Current Best Avg. Dice: {} Current Avg. Dice: {}".format(dice_val_best, dice_val)
                    )
            torch.cuda.empty_cache()

if __name__ == '__main__':
    main(params=params)