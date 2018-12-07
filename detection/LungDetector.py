# the standalone script to identify and classfy_nodel from Ningbo Data
# full_prep
from preprocessing import full_prep # preprocessing
from config_submit import config as config_submit # input
import glob # file processing Why not use os ?
import torch # structure for deep learning on GPU 
from torch.nn import DataParallel # paraller compute
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch import optim
from torch.autograd import Variable
from layers import acc

from data_detector import DataBowl3Detector,collate
from data_classifier import DataBowl3Classifier
import pandas as pd
from utils import *
from split_combine import SplitComb
from nodule_detect import nodule_detect
from importlib import import_module # import module
import pandas
import numpy as np
from scipy.special import expit
import pydicom
from layers import nms, iou, acc
import matplotlib.pyplot as plt
from preprocessing.step1 import load_scan, get_pixels_hu
datapath = config_submit['datapath']
prep_result_path = config_submit['preprocess_result_path']

skip_prep = True
skip_detect = True
print('Skip_prep',skip_prep)
print('Skip_detect',skip_detect)

print('Prep: Batch process a floder contains new patients (could be more than one), save _clean, _label, _spacing and _box into prep_result folder')
print('Detect: Using ./model detector ckpt to detetor lung nodule and save raw neural net results into ./bbox_results as _pbb')
print('Prep and detect can be skiped in a second run, change variable skip_prep and skip_detect into True')


if not skip_prep:
    print('It will take ups to 2min to do the preprocess, you will be able to skip it in a second run')
    testsplit = full_prep(datapath, prep_result_path,
                          n_worker = config_submit['n_worker_preprocessing'],
                          use_existing=config_submit['use_exsiting_preprocessing'])
    #print('region of interest shape:',sliceim.shape)
    #print('extendbox shape defined:')
    #print(extendbox)
else:
    print('Skip preprocessing')
    testsplit = os.listdir(datapath)
print('Loading QED nodule detection Master')
nodmodel = import_module(config_submit['detector_model'].split('.py')[0])
config1, nod_net, loss, get_pbb = nodmodel.get_model()
checkpoint = torch.load(config_submit['detector_param']) # ???
nod_net.load_state_dict(checkpoint['state_dict']) # load the trained net 

#torch.cuda.set_device(0)
#nod_net = nod_net.cuda()
#cudnn.benchmark = True
# Try working with CPU

nod_net = DataParallel(nod_net)

bbox_result_path = './bbox_result'
if not os.path.exists(bbox_result_path):
    os.mkdir(bbox_result_path)
testsplit = [f.split('_clean')[0] for f in os.listdir(prep_result_path) if '_clean' in f]
config1['crop_size'] = [32,32,32]
if not skip_detect:
    margin = 32
    sidelen = 64
    config1['datadir'] = prep_result_path
    split_comber = SplitComb(sidelen,config1['max_stride'],config1['stride'],margin,pad_value = config1['pad_value'])
    # what's SplitComb?
    dataset = DataBowl3Detector(testsplit,config1,phase='test',split_comber=split_comber)
    test_loader = DataLoader(dataset,batch_size = 1,
        shuffle = False,num_workers = 8,pin_memory=False,collate_fn =collate)

    nodule_detect(test_loader, nod_net, get_pbb, bbox_result_path,config1,n_gpu=config_submit['n_gpu'])


## coordinate mapping from resampled and box bounded data back into original dicom data


#df = pd.DataFrame(pbb)
# plot the nodule in plots
patientlist = glob.glob(config_submit['preprocess_result_path']+'*clean.npy') # get all *clean.npy files
patient_short = []
for patient in patientlist:
    patient_short.append(patient.split('_')[-2].split('/')[-1]) # get the prefix number
## save detector results for viewing
for save_id in range(len(patient_short)):
    df = []
    #print(save_id)
    patient = patient_short[save_id]
    pbbdir = './bbox_result/'+patient+'_pbb.npy'
    lbbdir = './bbox_result/'+patient+'_lbb.npy'
    pbb = np.load(pbbdir)
    lbb = np.load(lbbdir)
## the patient list here ## where you can change the directory id
    dicom_dir = config_submit['datapath']+patient
    case = load_scan(dicom_dir)
    case_pixels, spacing = get_pixels_hu(case) # what is spacing and case_pixels
    boundbox = np.load(config_submit['preprocess_result_path']+patient+'_box.npy')
    image_clean = np.load(config_submit['preprocess_result_path']+patient+'_clean.npy')
    pbb2 = nms(pbb,nms_th = 0.05) # what is nms ?
    pbb2[:, 0] = expit(pbb2[:, 0]) # expit(x) = 1/(1+exp(-x))
    image_clean = image_clean[0]
    # pbb 2 probability
    cands = pbb2 # z y x
    cands_spacing = cands
    cands_spacing[:,1:4] = (boundbox[:,0]+pbb2[:,1:4])/spacing
    cands_spacing = np.append(cands_spacing,np.random.choice(a=[False, True], size=(len(cands_spacing[:,0]), 1), p=[0.5, 0.5]),axis =1)
    cands_spacing = np.append(cands_spacing,np.random.randn(len(cands_spacing[:,0]), 1)*255,axis =1)
    cands_spacing = np.append(cands_spacing,np.random.choice(a=[False, True], size=(len(cands_spacing[:,0]), 1), p=[0.5, 0.5]),axis =1)
    #print(pd.DataFrame(cands_spacing))
    df =pd.DataFrame(cands_spacing)
    #df_index = pd.DataFrame(columns=['probability', 'index_z', 'index_y','index_x','diameter','is upper lober','HU_mean','Clinical decision'])
    df.columns=['probability', 'index_z', 'index_y','index_x','diameter','is upper lober','HU_mean','Clinical decision']
    #df.columns=['probability', 'index_z', 'index_y','index_x','diameter','is upper lober','HU_mean','Clinical decision']
    df.to_csv('./predictions/'+patient_short[save_id]+'.csv', index=False)
    #print(df)
## choose one to dispay
display_id = 1;
print('The result presenting now is case id %d, change variable display_id to view other case, range from 0 to %d'%(display_id, len(patient_short)-1))
patient = patient_short[display_id]
pbbdir = './bbox_result/'+patient+'_pbb.npy'
lbbdir = './bbox_result/'+patient+'_pbb.npy'
pbb = np.load(pbbdir)
lbb = np.load(lbbdir)


#dicom dir
dicom_dir = config_submit['datapath']+patient
print('Displaying detection from ', dicom_dir)
case = load_scan(dicom_dir)
case_pixels, spacing = get_pixels_hu(case)
boundbox = np.load(config_submit['preprocess_result_path']+patient+'_box.npy')
image_clean = np.load(config_submit['preprocess_result_path']+patient+'_clean.npy')
pbb2 = nms(pbb,nms_th = 0.05)
pbb2[:, 0] = expit(pbb2[:, 0])
image_clean = image_clean[0]
# pbb 2 probability
cands = pbb2 # z y x
cands_spacing = cands
cands_spacing[:,1:4] = (boundbox[:,0]+pbb2[:,1:4])/spacing
for voxelCoord in cands_spacing:
    #worldCoord = np.asarray([float(cand[3]),float(cand[2]),float(cand[1])])
    #voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
    #patch = numpyImage[int(voxelCoord[1]),iresultsnt(voxelCoord[2]-voxelWidth/2):int(voxelCoord[2]+voxelWidth/2),
    #                   int(voxelCoord[1]-voxelWidth/2):int(voxelCoord[1]+voxelWidth/2)]
    #patch = normalizePlanes(patch)
    #plt.imshow(patch, cmap='gray')
    plt.imshow(case_pixels[int(voxelCoord[1])],cmap = 'gray')
    ax = plt.gca()
    #plot using patch centre;
    radi = float(voxelCoord[4])/2/spacing[2]+1
    circle = plt.Circle((voxelCoord[3],voxelCoord[2]),int(radi),fill = False, color = 'r')
    #circle=plt.Circle((voxelWidth/2+1,voxelWidth/2+1),int(radi),fill = False,color = 'r')
    ax.add_artist(circle)
    plt.show()
