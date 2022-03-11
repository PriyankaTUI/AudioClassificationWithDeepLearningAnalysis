import numpy as np
import torch
import models
import copy
import torch.nn.functional as F
from dataset import MetaAudioDataLoader


class transfer_learning(object):
    def __init__(self, pretrained_model_path, optimizer = 'Adam', num_epochs = 10):
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # load model
        print("Loading the pre-trained model...")
        self.model = models.cnnModel()
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.model.load_state_dict(copy.deepcopy(torch.load(pretrained_model_path, map_location='cpu')), strict=False)
        # Loss function and optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def evaluate_model(self, X_test, y_test):
        """ Evaluate the model's performance on the test set X_test and y_test

        Parameters:
        ----------
        X_test: torch tensor of MFCC
        y_test: torch tensor of labels

        Returns:
        -------
        accuracy: average accuracy of the model
        accuracies: array of accuracy for each class
        """
        accuracy = 0  # average accuracy of the model
        accuracies = np.zeros(10)  # accuracy for each class
        nb_occurences = np.zeros(10)  # to convert counts to accuracy

        with torch.no_grad():
            for i in range(len(X_test)):
                prediction = self.model.predict(X_test[i].unsqueeze(0).to(device=self.device, dtype=torch.float))
                label = (int)(y_test[i].to(device= self.device).item())
                if prediction == label:
                    accuracies[label] = accuracies[label] + 1
                    accuracy = accuracy + 1
                nb_occurences[label] = nb_occurences[label] + 1

        accuracy = accuracy / len(X_test)
        accuracies = np.divide(accuracies, nb_occurences)

        return accuracy, accuracies

    def cl_train(self, inputs, labels):
        best_accuracy = 0.0
        print("Started adaptation of a model !")
        zipped = zip(inputs, labels)
        for i, (input, label) in enumerate(zipped):
            input = input.unsqueeze(1).float().to(self.device)
            label = label.unsqueeze(0).long().to(self.device)
            for epoch in range(self.num_epochs):
                self.model.train()
                running_loss = 0.0
                self.optimizer.zero_grad()
                outputs = self.model(input)
                loss = F.cross_entropy(outputs, label)
                loss.backward()
                self.optimizer.step()

    def test(self, inputs, labels):
        self.accuracy, self.accuracies = self.evaluate_model(inputs, labels)
        return


if __name__ == '__main__':
    FILE_PATH = "./savedmodels/best_model_state.pt"
    new_training_samples_path = "./dataset/data/sample_input_audio.pt"

    batch = torch.load(new_training_samples_path)
    inputs = torch.cat(batch, dim=0)
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    labels = torch.tensor(digits)

    # test for different outputs:
    meta_dataloader = MetaAudioDataLoader(10, 0.2)
    test_inputs, test_labels = meta_dataloader.get_test_samples(5)
    # Analysis for Adam tranfer learning
    tl_adam = transfer_learning(pretrained_model_path=FILE_PATH, num_epochs=5)
    tl_adam.cl_train(inputs, labels)
    tl_adam.test(test_inputs, test_labels)

    print(tl_adam.accuracy)
    print(tl_adam.accuracies)