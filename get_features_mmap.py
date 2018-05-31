import os
import numpy as np
import pickle
import re
import multiprocessing as mp
import logging

from collections import Counter
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize, sent_tokenize
from tqdm import tqdm
from PIL import Image
import pdb
from ipdb import slaunch_ipdb_on_exception
import gc

logger = logging.getLogger("Data preprocessing")
logger.setLevel(logging.INFO)
logging.basicConfig(format='%(levelname)s %(asctime)s : %(message)s', level=logging.INFO)

FILE_BASE= '_img_features.dat'
BATCH_SIZE= 100
NUM_WORKERS= 8

class CocoCaptions_Cust(Dataset):
    def __init__(self, root, annFile, transform=None, target_transform=None):
        from pycocotools.coco import COCO
        self.root = os.path.expanduser(root)
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        #pdb.set_trace()
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        target = [ann['caption'] for ann in anns]

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, img_id

    def __len__(self):
        return len(self.ids)


#download pre-trained models
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet34 = models.resnet34(pretrained=True)
resnet50 = models.resnet50(pretrained=True)
resnet101 = models.resnet101(pretrained=True)
resnet152 = models.resnet152(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
inception = models.inception_v3(pretrained=True)


def run_cnn_models(model_dict, model_name):
    print("Model is {}".format(model_name))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model= model_dict[model_name]
    model_str= model_name
    model= model.to(device)
    model.eval()
    path= '/home/hanozbhathena/project/data/img_features/' + model_str
    file= model_str + FILE_BASE
    train_outfilename= os.path.join(path, 'train_' + file)
    val_outfilename= os.path.join(path, 'val_' + file)    
    
    #Prepare the dataset class

    data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    train_caption_dset = CocoCaptions_Cust(root= '/home/hanozbhathena/project/data/resizetrain2017', 
                                     annFile= '/home/hanozbhathena/project/data/annotations/captions_train2017.json',
                                     transform= data_transform)

    train_dataloader= DataLoader(train_caption_dset, batch_size= BATCH_SIZE,
                                   shuffle=False, num_workers= NUM_WORKERS)

    val_caption_dset = CocoCaptions_Cust(root= '/home/hanozbhathena/project/data/resizeval2017', 
                                     annFile= '/home/hanozbhathena/project/data/annotations/captions_val2017.json',
                                     transform= data_transform)

    val_dataloader= DataLoader(val_caption_dset, batch_size= BATCH_SIZE,
                               shuffle=False, num_workers= NUM_WORKERS)


    #Training set loop
    test= model(train_caption_dset[0][0].unsqueeze(0).to(device)).detach().cpu().numpy()
    img_class_distr= np.memmap(train_outfilename, dtype= np.float32, mode= 'w+', shape= (len(train_caption_dset), *test.shape[1:]))
    with torch.no_grad():
        for i, (img, _, _) in enumerate(train_dataloader):
            img= img.to(device)
            distr= model(img)
            img_class_distr[i*BATCH_SIZE:min(len(img_class_distr), (i+1)*BATCH_SIZE)]= distr.cpu().numpy()
            if i > 0 and i*BATCH_SIZE % 1000 == 0:
                logging.info("Train Finished {}/{}".format(i*BATCH_SIZE, len(train_caption_dset)))

    # pdb.set_trace()
    print(img_class_distr.shape)
    del img_class_distr
    
    gc.collect()

    #Dev set loop
    img_class_distr= np.memmap(val_outfilename, dtype= np.float32, mode= 'w+', shape= (len(val_caption_dset), *test.shape[1:]))
    with torch.no_grad():
        for i, (img, _, _) in enumerate(val_dataloader):
            img= img.to(device)
            distr= model(img)
            data= distr.cpu().numpy()
            img_class_distr[i*BATCH_SIZE:min(len(img_class_distr), (i+1)*BATCH_SIZE)]= data
            if i > 0 and i*BATCH_SIZE % 1000 == 0:
                logging.info("Val Finished {}/{}".format(i*BATCH_SIZE, len(val_caption_dset)))

    # pdb.set_trace()
    print(img_class_distr.shape)
    del img_class_distr
    logging.info("{} done".format(model_name))
    
    

def get_model_wo_last_n_layers(model, n= 1):
    if n == 0:
        return model
    layers= []
    for name, layer in model._modules.items():
        layers.append(layer)
    return_model= torch.nn.Sequential(*layers[:-n])
    return return_model


if __name__ == "__main__":
    with slaunch_ipdb_on_exception():

        model_dict= {'resnet18': resnet18,
                    'resnet34': resnet34,
                    'resnet50': resnet50,
                    'resnet101': resnet101,
                    'resnet152': resnet152,
                    'alexnet': alexnet,
                    'squeezenet': squeezenet,
                    'vgg16': vgg16,
                    'densenet': densenet,
                    'inception': inception,
                    }

        models_to_run= ['alexnet', 'squeezenet', 'resnet18']

        for model_name in models_to_run:
            model_dict[model_name]= get_model_wo_last_n_layers(model_dict[model_name])
            run_cnn_models(model_dict, model_name)

