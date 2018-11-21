import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

class SimpleGridNoLongiDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.fnames = sorted(make_dataset(os.path.join(self.dir, 'topo'), fnameOnly=True))

    def __getitem__(self, index):
        fname = self.fnames[index]

        w_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, self.opt.loadSize - self.opt.fineSize - 1))

        topo = Image.open(os.path.join(self.dir, 'topo', fname))
        topo = topo.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        topo = transforms.ToTensor()(topo)
        topo = topo[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        topo = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(topo)

        land = Image.open(os.path.join(self.dir, 'land', fname))
        land = land.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        land = transforms.ToTensor()(land)
        land = land[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        land = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(land)
    
        lati = Image.open(os.path.join(self.dir, 'lati', fname))
        lati = lati.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        lati = Image.fromarray((np.asarray(lati) / 257).astype(np.uint8))
        lati = transforms.ToTensor()(lati)
        lati = lati[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
        lati = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lati)

        A = torch.cat((topo, land, lati), dim=0)

        if self.opt.phase == 'train':
            bm = Image.open(os.path.join(self.dir, 'bm', fname))
            bm = bm.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
            bm = transforms.ToTensor()(bm)
            bm = bm[:, h_offset:h_offset + self.opt.fineSize, w_offset:w_offset + self.opt.fineSize]
            bm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(bm)
            B = bm       

        else:
            B = np.zeros_like(topo)  

        return {'A': A, 'B': B,
                'A_paths': fname, 'B_paths': fname}

    def __len__(self):
        return len(self.fnames)

    def name(self):
        return 'SimpleGridNoLongiDataset'
