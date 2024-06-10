from collections import defaultdict
import torch, random
import numpy as np       

class Masking(object):
    def __init__(self, alpha, masking_constant=0.):
        self.mask_prob = np.random.beta(alpha, alpha)
        if type(masking_constant) == str:
            self.masking_constant = eval(masking_constant)
        else:
            self.masking_constant = masking_constant
    
    def __call__(self, sample):
        img = sample['image']
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob    
        img[mask] = self.masking_constant
        return {'image': img, 'mask': torch.tensor(mask, device='cuda').requires_grad_(requires_grad=False)}

class Shuffling(object):
    def __init__(self, alpha):
        self.mask_prob = np.random.beta(alpha, alpha)
        self.seed = random.randint(0, 99999)
    
    def __call__(self, sample):
        img = sample['image'].to('cuda')
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device='cuda')
        
        permuted = torch.empty(size=img.size()).to('cuda')
        for f in range(img.size(1)):
            permuted[:, f] = img[torch.randperm(img.size(0)), f]

        return {'image': img * (1-mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask}

    
class BinShuffling(object):
    def __init__(self, alpha, bin_interval, num_bins):
        self.mask_prob = np.random.beta(alpha, alpha)
        self.seed = random.randint(0, 99999)
        self.bin_interval = bin_interval
        self.num_bins = num_bins
        
        
    def __call__(self, sample):
        img = sample['image'].to('cuda')
        mask = np.random.uniform(0, 1, size=img.shape) < self.mask_prob
        mask = torch.tensor(mask, device='cuda')
        
        permuted = torch.empty(size=img.size()).to('cuda')
        
        if self.num_bins == 1:
            for f in range(img.size(1)):
                permuted[:, f] = img[torch.randperm(img.size(0)), f]

            return {'image': img * (1-mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask}
        else:
            ranks = sample['ranks'] // self.bin_interval
            for f in range(img.size(1)):
                upper_resid = torch.where(ranks >= self.num_bins[f])
                for (x, y) in zip(upper_resid[0], upper_resid[1]):
                    ranks[x, y] = (self.num_bins[f] - 1)
                for b in range(self.num_bins[f]):
                    idx = torch.where(ranks[:, f] == b)[0]
                    perm_idx = idx[torch.randperm(len(idx))]
                    permuted[idx, f] = img[perm_idx, f]

            return_img = img * (1 - mask.type(torch.int)) + permuted * mask.type(torch.int)

            return {'image': img * (1 - mask.type(torch.int)) + permuted * mask.type(torch.int), 'mask': mask}
    
    
class ToTensor(object):
    def __call__(self, sample):
        if isinstance(sample['image'], np.ndarray):
            return {'image': torch.from_numpy(sample['image']), 'mask': torch.from_numpy(sample['mask'])}
        else:
            return {'image': sample['image'], 'mask': sample['mask']}

        