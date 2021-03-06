import os # manipulate files
import numpy as np # import numpy lib
from scipy.io import loadmat # import scipy lib for more complex math operations 
# load .mat file  It does not be used ?? 
import h5py # used to create h5py files
from scipy.ndimage.interpolation import zoom # Zoom an array
from skimage import measure #skimage is a photo processing lib
import warnings
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from skimage.morphology import convex_hull_image # compute convex part
from multiprocessing import Pool # multi process pool provide number of process 
from functools import partial # redecorate function
from step1 import step1_python
import warnings

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1  = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1)>0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2)>2*np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1) # generate structional element 
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10) 
    return dilatedMask

# def savenpy(id):
id = 1

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def resample(imgs, spacing, new_spacing, order = 2):
    if len(imgs.shape)==3:
        new_shape = np.round(imgs.shape * spacing / new_spacing)
        true_spacing = spacing * imgs.shape / new_shape
        resize_factor = new_shape / imgs.shape
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            imgs = zoom(imgs, resize_factor, mode = 'nearest', order=order)
        return imgs, true_spacing
    elif len(imgs.shape)==4:
        n = imgs.shape[-1]
        newimg = []
        for i in range(n):
            slice = imgs[:,:,:,i]
            newslice,true_spacing = resample(slice,spacing,new_spacing)
            newimg.append(newslice)
        newimg=np.transpose(np.array(newimg),[1,2,3,0])
        return newimg,true_spacing
    else:
        raise ValueError('wrong shape')

def savenpy(id, filelist, prep_folder, data_path, use_existing=True):      
    resolution = np.array([1, 1, 1])
    name = filelist[id]
    if use_existing:
        if os.path.exists(os.path.join(prep_folder,name+'_label.npy')) and os.path.exists(os.path.join(prep_folder,name+'_clean.npy')):
            print(name+' had been done')
            return
    try:
        im, m1, m2, spacing = step1_python(os.path.join(data_path,name))
        Mask = m1+m2
        
        newshape = np.round(np.array(Mask.shape)*spacing/resolution)
        xx,yy,zz= np.where(Mask)
        box = np.array([[np.min(xx),np.max(xx)],[np.min(yy),np.max(yy)],[np.min(zz),np.max(zz)]])
        box = box*np.expand_dims(spacing,1)/np.expand_dims(resolution,1)
        box = np.floor(box).astype('int')
        margin = 5
        extendbox = np.vstack([np.max([[0,0,0],box[:,0]-margin],0),np.min([newshape,box[:,1]+2*margin],axis=0).T]).T
        extendbox = extendbox.astype('int')

        convex_mask = m1
        dm1 = process_mask(m1)
        dm2 = process_mask(m2)
        dilatedMask = dm1+dm2
        Mask = m1+m2
        extramask = dilatedMask ^ Mask
        bone_thresh = 210
        pad_value = 170

        im[np.isnan(im)]=-2000
        sliceim = lumTrans(im)
        sliceim = sliceim*dilatedMask+pad_value*(1-dilatedMask).astype('uint8')
        bones = sliceim*extramask>bone_thresh
        sliceim[bones] = pad_value
        # the lung greater ROI is extracted from resampled image volue (1*1*1) 
        sliceim1,_ = resample(sliceim, spacing, resolution, order=1)
        sliceim2 = sliceim1[extendbox[0,0]:extendbox[0,1],
                    extendbox[1,0]:extendbox[1,1],
                    extendbox[2,0]:extendbox[2,1]]
        sliceim = sliceim2[np.newaxis,...]
        print('Saving preprocessing result into ./prep_results')
        np.save(os.path.join(prep_folder,name+'_clean'),sliceim)
        np.save(os.path.join(prep_folder,name+'_label'),np.array([[0,0,0,0]]))
        np.save(os.path.join(prep_folder,name+'_box'),extendbox)
        np.save(os.path.join(prep_folder,name+'_spacing'),spacing)
        return sliceim, extendbox
	# we dont save that anylonger and we want the extended box returned as a varaiable so we can map the coordinated back into dicom fomat        
    # ok i have change my mind, we still need to find save that beacause we might want process multiple patient in future.
    except:
        print('bug in '+name)
        raise
    print(name+' done')

    
def full_prep(data_path,prep_folder,n_worker = None,use_existing=True):
    # the old version deals with batch processing and we dont actually need this
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)

            
    ##print('starting preprocessing')
    ##filelist = [f for f in os.listdir(data_path)]
    ##id = 0; # say we only have 1 patient in the datapath to be processed
    ##sliceim, extendbox = savenpy(id, filelist=filelist,prep_folder=prep_folder, data_path=data_path,use_existing=use_existing)
    ##return filelist, sliceim, extendbox

    #sliceim, extendbox = savenpy()
    #partial_savenpy = partial(savenpy,filelist=filelist,prep_folder=prep_folder,
    #                          data_path=data_path,use_existing=use_existing)

    # The preprocessing takes upto 2 secs to build the image;
    #N = len(filelist)
    #_=pool.map(partial_savenpy,range(N))
    #pool.close()
    #pool.join()

    # parallel processing
    warnings.filterwarnings("ignore")
    if not os.path.exists(prep_folder):
        os.mkdir(prep_folder)


    print('starting preprocessing')
    pool = Pool(n_worker) # n_worker = 8
    filelist = [f for f in os.listdir(data_path)] 
    partial_savenpy = partial(savenpy, filelist=filelist, 
                              prep_folder=prep_folder, data_path=data_path, 
                              use_existing=use_existing)

    N = len(filelist)
    _=pool.map(partial_savenpy,range(N))
    pool.close()
    pool.join()
    print('end preprocessing')
    return filelist
