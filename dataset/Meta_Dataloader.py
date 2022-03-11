import numpy as np
import memory_profiler as mem_profile
import matplotlib.pyplot as plt
import torch

class MetaAudioDataLoader(object):
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    def __init__(self , k_shot, test_size):
        self.n_way = 10 #we have 10 digits with different class
        self.k_shot = int(k_shot)
        self.test_size = test_size
        self.task_num = len(self.digits)

        # print('Memory before : {}MB'.format(mem_profile.memory_usage()))
        #todo: change directory path with more generic case
        self.X = np.load("./dataset/data/X.npy")
        y = np.load("./dataset/data/y.npy")
        self.task_indices = [np.where(y==i) for i in self.digits]#{label1: index_audio1, index_audio2, ....; label2: index_audio1, index_audio2, ...}
        # print('Memory before task index  : {}MB'.format(mem_profile.memory_usage()))

    def cache_data(self, task_id):
        support_size = self.n_way
        query_size = int(support_size * self.test_size)
        # x_support , x_querry = [], []
        samples_indices = np.random.choice((self.task_indices[task_id][0]), support_size +  query_size)
        x_support = self.X[samples_indices[0:support_size]]
        x_querry = self.X[samples_indices[support_size:]]
        x_supt = torch.from_numpy(x_support)
        x_quer = torch.from_numpy(x_querry)
        return [x_supt , x_quer]

    def cache_validation_set(self):
        x_test = []
        y_test = []
        for i in self.digits:
            samples_indices = np.random.choice((self.task_indices[i][0]), 1)
            x_test.append(self.X[samples_indices])
            y_test.append(i)

        return torch.FloatTensor(x_test), torch.FloatTensor(y_test)

    def get_samples(self, task_id):
        samples_indices = np.random.choice((self.task_indices[task_id][0]), self.k_shot)
        x_support = self.X[samples_indices]
        samples = torch.from_numpy(x_support)
        return samples

    def get_test_samples(self, batch_size):
        inputs = []
        labels = []
        for _ in range(batch_size):
            for i in range(len(self.digits)):
                samples_indices = np.random.choice((self.task_indices[i][0]), 1)
                input = self.X[samples_indices]
                inputs.append(input)
                labels.append(self.digits[i])
        inputs = torch.tensor(inputs)
        labels = torch.tensor(labels)
        return inputs, labels
