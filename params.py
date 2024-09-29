import os
import datetime

params = dict()

params['base_dir'] = './outputs/tmp'
params['model_name'] = './pretrained/model.pth'
params['data_dir'] = './inputs/tmp/'
params['model_type'] = 'MS_DSA_NET' 
params['sa_type'] = 'parallel'
params['chans_in']   = 2
params['chans_out']  = 2
params['feature_size'] = 16 
params['project_size'] = 64 #dsa projection size
params['patch_size'] = [128]*3 
params['num_workers'] = 0
params['seq'] = 't1+t2'
params['chans_in'] = 2
params['date'] = datetime.date.today()

os.makedirs(params['base_dir'], exist_ok=True)