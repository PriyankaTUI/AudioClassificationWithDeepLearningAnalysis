import torch
import matplotlib.pyplot as plt
import librosa
from statistics import mean
import torch.nn.functional as F
import copy

def label_to_index(labels, label):
    # Return the position of the word in labels
    return torch.tensor(labels.index(label))

def index_to_label(labels, index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]

def split_batch(inputs, labels):
    #split batch in two sub set to use in meta learning 
    support_inputs, query_inputs = inputs.chunk(2, dim=0)
    support_labels, query_labels = labels.chunk(2, dim=0)
    return support_inputs, support_labels, query_inputs, query_labels


def get_accuracy(logits, labels):
        _, pred = logits.max(1)
        return (pred == labels).sum().item() / pred.size(0)


def get_average_of_list(lst):
    return sum(lst)/len(lst)

def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)
    return

def calculate_accuracy(dataloader, model, device = 'cpu'):
  #Accuracy for given test data
    model.to(device)
    with torch.no_grad():
      class_acc =[]
      model.eval()
      for i, (inputs, labels) in enumerate(dataloader):
        logits = model(inputs)
        loss = F.cross_entropy(logits, labels)
        _, pred = logits.max(1)
        acc = (pred == labels).sum().item() / pred.size(0)
        class_acc.append(acc)
    return mean(class_acc), loss.item()


# def get_convnet(convnet_type):
#     name = convnet_type.lower()
#     if name == "vgg":
#       FILE_PATH = "./savedmodels/vgg_checkpoint.pth"
#       print("getting VGG pre-trained model")
#       vgg_model = models.VGGNet()
#       vgg_model.load_state_dict(copy.deepcopy(torch.load(FILE_PATH, map_location='cpu')), strict=False)
#       return vgg_model
#     elif name == "alexnet":
#       FILE_PATH = "./savedmodels/checkpoint_alexnet.pth"
#       print("getting AlexNet pre-trained model")
#       alexnet_model = models.AlexNet()
#       alexnet_model.load_state_dict(copy.deepcopy(torch.load(FILE_PATH, map_location='cpu')), strict=False)
#       return alexnet_model