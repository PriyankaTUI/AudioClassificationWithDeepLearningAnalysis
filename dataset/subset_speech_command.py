from torchaudio.datasets import SPEECHCOMMANDS
import os
import torchaudio
import numpy as np
from operator import itemgetter
# import torch
# from torch.utils.data import Dataset, DataLoader
# from collections import defaultdict
# import random

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, 
                subset: str = None, 
                subset_type : str = None, 
                novel_class_list:list = [], 
                dataset_length:int = 0,
                transform = None):
        digits = ['zero','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'] 
        super().__init__("./dataset/data/", download=True)
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256
        n_mfcc = 256
        sampling_rate = 16000
        if transform == None:
            self.transform = torchaudio.transforms.MFCC(sample_rate=sampling_rate, n_mfcc=32, 
                                                                            melkwargs={
                                                                                        'n_fft': n_fft,
                                                                                        'n_mels': n_mels,
                                                                                        'hop_length': hop_length,
                                                                                        'mel_scale': 'htk',
                                                                                        })
        else:
            self.transform = transform

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "training":
            _exclude = load_list("testing_list.txt") + load_list("validation_list.txt")
            if subset_type == 'old':
                self._walker = [w for w in self._walker if w is not _exclude and os.path.basename(os.path.dirname(w)) in digits]
            elif subset_type == 'novel':
                if novel_class_list:
                    self._walker = [w for w in self._walker if w is not _exclude and os.path.basename(os.path.dirname(w)) not in digits and os.path.basename(os.path.dirname(w)) in novel_class_list]
                else:
                    self._walker = [w for w in self._walker if w is not _exclude and os.path.basename(os.path.dirname(w)) not in digits]
        elif subset == "testing": 
            _include = load_list("testing_list.txt")
            if subset_type == 'old':
                self._walker = [w for w in _include if os.path.basename(os.path.dirname(w)) in digits]
            elif subset_type == 'novel':
                if novel_class_list:
                    self._walker = [w for w in _include if os.path.basename(os.path.dirname(w)) not in digits and os.path.basename(os.path.dirname(w)) in novel_class_list]
                else:
                    self._walker = [w for w in _include if os.path.basename(os.path.dirname(w)) not in digits]
        elif subset == "validation": 
            _include = load_list("validation_list.txt")
            if subset_type == 'old':
                self._walker = [w for w in _include if os.path.basename(os.path.dirname(w)) in digits]
            elif subset_type == 'novel':
                if novel_class_list:
                    self._walker = [w for w in _include if os.path.basename(os.path.dirname(w)) not in digits and os.path.basename(os.path.dirname(w)) in novel_class_list]
                else:
                    self._walker = [w for w in _include if os.path.basename(os.path.dirname(w)) not in digits]


        if dataset_length:
            random_indices = np.random.randint(len(self._walker) ,size = dataset_length)
            self._walker = list(itemgetter(*random_indices)(self._walker))
            
    def __getitem__(self,idx):
        waveform, sampling_rate, label, *_ = super().__getitem__(idx)
        #returning waveform and it's label
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

    # def __len__(self) -> int:
    #     return 50