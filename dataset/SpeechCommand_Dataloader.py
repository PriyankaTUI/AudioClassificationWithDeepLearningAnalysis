import torch
import torchaudio
import os
from torch.utils.data import Dataset

def label_to_index(labels, label):
    # Return the position of the word in labels
    return torch.tensor(labels.index(label))

def index_to_label(labels, index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def load_and_preprocess_speech_command_dataset(random_targets, digits):
    tensors = []
    targets = []

    #old classes storage
    old_class_tensors = []
    old_class_targets = []
    
    labels = digits + random_targets
    # print(f"Novel classes: {random_targets}")
    # print(f"Old classes: {digits}")
    # print(f"List of all classes: {labels}")

    ### saving local file for some data to save time for creating and processing new data
    ### while continuously working on project 
    ### we can delete local files and always create new data
    if (os.path.exists(path='dataset/data/novel_class_tensors.pt') and 
        os.path.exists(path='dataset/data/novel_class_targets.pt') and 
        os.path.exists(path='dataset/data/old_class_tensors.pt') and 
        os.path.exists(path='dataset/data/old_class_targets.pt')):
        
        tensors = torch.load('dataset/data/novel_class_tensors.pt')
        targets = torch.load('dataset/data/novel_class_targets.pt')
        old_class_tensors = torch.load('dataset/data/old_class_tensors.pt')
        old_class_targets = torch.load('dataset/data/old_class_targets.pt')

    else:

        #  Loading dataset and custom dataloader
        dataset = torchaudio.datasets.SPEECHCOMMANDS('./dataset/data/' , url = 'speech_commands_v0.02', folder_in_archive= 'SpeechCommands',  download = True)
        #parameters for MFCC transformation
        n_fft = 2048
        win_length = None
        hop_length = 512
        n_mels = 256
        n_mfcc = 256

        for waveform, sample_rate, label, *_ in dataset:
            if label in random_targets:
                if sample_rate == 16000:
                    if waveform.shape == (1, 16000):
                        tensors += [torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=32, 
                                                                melkwargs={
                                                                            'n_fft': n_fft,
                                                                            'n_mels': n_mels,
                                                                            'hop_length': hop_length,
                                                                            'mel_scale': 'htk',
                                                                            }
                                                                            )(waveform)]
                        targets += [label_to_index(labels, label)]

                if label in digits:
                    if sample_rate == 16000:
                        if waveform.shape == (1, 16000):
                            old_class_tensors += [torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=32, 
                                                                    melkwargs={
                                                                                'n_fft': n_fft,
                                                                                'n_mels': n_mels,
                                                                                'hop_length': hop_length,
                                                                                'mel_scale': 'htk',
                                                                                }
                                                                                )(waveform)]
                            old_class_targets += [label_to_index(labels, label)]

        torch.save(tensors, 'dataset/data/novel_class_tensors.pt')
        torch.save(targets, 'dataset/data/novel_class_targets.pt')
        torch.save(old_class_tensors, 'dataset/data/old_class_tensors.pt')
        torch.save(old_class_targets, 'dataset/data/old_class_targets.pt')

    return {"tensors": tensors, "targets": targets,
            "old_class_tensors": old_class_tensors, 
            "old_class_targets": old_class_targets}
    


class SpeechCommandSubDataset(Dataset):
    
    def __init__(self,data,labels):
        self.data = data
        self.labels = labels
            
    def __len__(self):
        return len(self.data)    
    
    def __getitem__(self,idx):
        # print(f"getting data {idx}")
        return self.data[idx], self.labels[idx]
