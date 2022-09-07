import torch

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