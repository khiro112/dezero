if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    
import numpy as np

class Dataset:
    def __init__(self, train=True, transform=None, target_transform=None):
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        if self.transform is None:
            self.transform = lambda x: x
        if self.target_transform is None:
            self.target_transform = lambda x: x
            
        self.data = None
        self.label = None
        self.prepare()
        
    def __getitem__(self, index):
        assert np.isscalar(index)
        if self.label is None:
            return self.transform(self.data[index]), None
        else:
            return self.transform(self.data[index]),\
                   self.target_transform(self.label[index])
        
    def __len__(self)
        return len(self.data)    
        
    def prepare(self):
        pass
    
class Spiral(Dataset):
    def prepare(self):
        self.data, self.label = get_spiral(self.train)
        
