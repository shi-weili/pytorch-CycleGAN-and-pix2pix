import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image


class SimpleGridDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.fnames = sorted(make_dataset(os.path.join(self.dir, 'topo'), fnameOnly=True))

    def __getitem__(self, index):
        fname = self.fnames[index]

        topo = Image.open(os.path.join(self.dir, 'topo', fname))
        topo = topo.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        topo = transforms.ToTensor()(topo)
        topo = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(topo)

        land = Image.open(os.path.join(self.dir, 'land', fname))
        land = land.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        land = transforms.ToTensor()(land)
        land = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(land)

        longi = Image.open(os.path.join(self.dir, 'longi', fname))
        longi = longi.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        longi = transforms.ToTensor()(longi).type(torch.FloatTensor) / 64800
        longi = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(longi)
    
        lati = Image.open(os.path.join(self.dir, 'lati', fname))
        lati = lati.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
        lati = transforms.ToTensor()(lati).type(torch.FloatTensor) / 64800
        lati = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lati)

        A = torch.cat((topo, land, longi, lati), dim=0)

        if self.opt.phase == 'train':
            bm = Image.open(os.path.join(self.dir, 'bm', fname))
            bm = bm.resize((self.opt.fineSize, self.opt.fineSize), Image.LANCZOS)
            bm = transforms.ToTensor()(bm)
            bm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(bm)
            B = bm       

        else:
            B = np.zeros_like(topo)   

        return {'A': A, 'B': B,
                'A_paths': fname, 'B_paths': fname}

    def __len__(self):
        return len(self.fnames)

    def name(self):
        return 'SimpleGridDataset'
