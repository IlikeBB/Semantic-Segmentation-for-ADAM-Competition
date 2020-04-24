import argparse
import os
import numpy as np
import scipy.io as io
import torch
import cv2, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

from dataloaders.utils import decode_segmap
MI = np.load('meanimage.npy')

numClass={'pascal':6,
'coco':21,
'cityscapes':19}
classes = ['drusen', 'hemorrhage', 'exudate', 'scar', 'others']
for cid, c in enumerate(classes):
    if not os.path.isdir('result_val1/'+c):
        os.mkdir('result_val1/' + c)
        
cuda = torch.cuda.is_available()
cuda = False
nclass = numClass['pascal']
model = DeepLab(num_classes=nclass, backbone='resnet', output_stride=16, sync_bn=None, freeze_bn=False)
weight_dict=torch.load('run/pascal/exp6/model_best.pth.tar', map_location='cpu')
if cuda:
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()
    
    model.module.load_state_dict(weight_dict['state_dict'])
else:
    model.load_state_dict(weight_dict['state_dict'])
model.eval()


filenames = glob.glob('valimg/*.jpg')
# filenames = glob.glob('valimg/*.jpg')
kernel = np.ones((5,5),np.uint8)
kernel2 = np.ones((11,11),np.uint8)
for imgPath in filenames:
    fn = os.path.basename(imgPath)[:-4]
#     outPath = 'result/' + fn +'.png'
    outPath = 'result_val1/' + fn +'.png'

    image = cv2.imread(imgPath)
    oriDim = image.shape
    image = cv2.resize(image, dsize=(513,513)) - MI
    image = image.astype(np.float32) / 255.
    image = image[:, :, ::-1]
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    
    
    for i in range(3):
        image[:, :, i] = image[:, :, i] - means[i]
        image[:, :, i] = image[:, :, i] / stds[i]

    image = torch.from_numpy(image.transpose((2, 0, 1)).astype(np.float32)).float().unsqueeze(0)

    if cuda:
        image = image.cuda()
        
    with torch.no_grad():
#         mat_path='pre_val/'+fn+'.mat' #1
        output = model(image)
        output=torch.sigmoid(output) #output shape torch.Size [1,6,513,513]
        
        output2 = output.data.cpu().numpy() #output2 shape [1,6,513,513]
#         io.savemat(mat_path,{fn:output2}) #1

        prediction = np.argmax(output2, axis=1)[0] #prediction[0][1] shape[513,513]
        

        ps=[]
        for cid, c in enumerate(classes): #c=class's name cid=channel+1 1~5
#             C1=output[0,cid+1] #C1 shape torch.Size [513,513]
#             C1[C1>0.7]=255
#             C1[C1<=0.7]=0
#             C2 = C1.data.cpu().numpy() #C2 shape [513,513]
            
            #        
            mask = np.zeros((prediction.shape[0], prediction.shape[1]), np.uint8) +255
            mask[prediction == cid+1] = 0 #mask shape [513,513]
            #            

            mask = cv2.morphologyEx(255-mask, cv2.MORPH_OPEN, kernel)
#             mask = cv2.morphologyEx(C2, cv2.MORPH_OPEN, kernel)
    
            mask = 255-cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel2)
            mask = cv2.resize(mask,dsize=(oriDim[1],oriDim[0]), interpolation=cv2.INTER_NEAREST)
            
            ps.append(np.mean(prediction == cid+1))

            cv2.imwrite('result_val1/' + c + '/' + fn+'.png', mask)
        segmap = decode_segmap(prediction, dataset='pascal')

        segmap = (segmap*255).astype(np.uint8)
        segmap = cv2.resize(segmap,dsize=(oriDim[1],oriDim[0]))
        segmap = segmap[:, :, ::-1]
        cv2.imwrite(outPath, segmap)
    print('Done inference '+fn,'percentage:', ps)
exit(1)