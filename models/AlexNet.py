import torch.nn as nn
import torch
from utils import initialization_constants
import copy


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # self.classifier = nn.Sequential(
        #     nn.Dropout(p=dropout),
        #     nn.Linear(256 * 6 * 6, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout(p=dropout),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, num_classes),
        # )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.linear1 = nn.Linear(256 * 6 * 6, out_features=1024)
        self.linear2 = nn.Linear(1024, 1024)
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dropout(x)
        x = self.linear1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear(x)
        # x = self.classifier(x)
        return x

    def freeze_old_params(self):
        # freeze all layers except last liner layer
        for names, param in self.named_parameters():
            if 'linear' not in names:
                param.requires_grad = False

    def unfreeze_old_params(self):
        # unfreeze all layers except last liner layer
        for param in self.parameters():
            param.requires_grad = True

    def addNovelClassesToModel(self, noNovelClasses, initialization=initialization_constants.RANDOM):
        # add new novel classes to pre-trained model
        # add new class parameters , here we have added 3 new classes
        input_features = self.linear.in_features
        output_features = self.linear.out_features

        # print("Old Model:")
        # summary(self,input_size=(1,32,32))

        # save last three linear layers (classifiers) weights for initialization step
        # old_weights_linear1 = copy.deepcopy(self.linear1.weight.data) 
        # old_bias_linear1 = copy.deepcopy(self.linear1.bias.data) 
        # old_weights_linear2 = copy.deepcopy(self.linear2.weight.data) 
        # old_bias_linear2 = copy.deepcopy(self.linear2.bias.data)
        old_weights_linear = copy.deepcopy(self.linear.weight.data)
        old_bias_linear = copy.deepcopy(self.linear.bias.data)

        new_output_features = output_features + noNovelClasses
        print(f"new output features: {new_output_features}")

        # self.linear1 = nn.Linear(in_features=input_features, out_features= new_output_features)
        # self.linear2 = nn.Linear(in_features=new_output_features, out_features= new_output_features)
        self.linear = nn.Linear(in_features=input_features, out_features=new_output_features)

        # Initialize new weights with kaiming normal distribution
        # in paper xaviour is suggested for relu kaiming is recommended
        if initialization == initialization_constants.KAIMING_UNIFORM:
            # nn.init.kaiming_uniform_(self.linear1.weight, nonlinearity='relu')
            # nn.init.kaiming_uniform_(self.linear2.weight, nonlinearity='relu')
            nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        elif initialization == initialization_constants.KAIMING_NORMAL:
            # nn.init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
            # nn.init.kaiming_normal_(self.linear2.weight, nonlinearity='relu')
            nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
        elif initialization == initialization_constants.KAIMING_UNIFORM_FAN_OUT:
            # nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_uniform_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        elif initialization == initialization_constants.KAIMING_NORMAL_FAN_OUT:
            # nn.init.kaiming_normal_(self.linear1.weight, mode='fan_out', nonlinearity='relu')
            # nn.init.kaiming_normal_(self.linear2.weight, mode='fan_out', nonlinearity='relu')
            nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        elif initialization == initialization_constants.XAVIER_UNIFORM:
            # nn.init.xavier_uniform_(self.linear1.weight)
            # nn.init.xavier_uniform_(self.linear2.weight)
            nn.init.xavier_uniform_(self.linear.weight)
        elif initialization == initialization_constants.XAVIER_NORMAL:
            # nn.init.xavier_normal_(self.linear1.weight)
            # nn.init.xavier_normal_(self.linear2.weight)
            nn.init.xavier_normal_(self.linear.weight)
        elif initialization == initialization_constants.ZEROS:
            # self.linear1.weight.data.fill_(0)
            # self.linear2.weight.data.fill_(0)
            self.linear.weight.data.fill_(0)
        elif initialization == initialization_constants.NEGATIVE:
            # self.linear1.weight.data.fill_(-1)
            # self.linear2.weight.data.fill_(-1)
            self.linear.weight.data.fill_(-1)
        # copy old weights for old paramerter
        # self.linear1.weight.data[:output_features] = old_weights_linear1
        # self.linear1.bias.data[:output_features] = old_bias_linear1
        # self.linear2.weight.data[:output_features] = old_weights_linear2
        # self.linear2.bias.data[:output_features] = old_bias_linear2
        self.linear.weight.data[:output_features] = old_weights_linear
        self.linear.bias.data[:output_features] = old_bias_linear

        input_features = self.linear.in_features
        output_features = self.linear.out_features
        print(f"New model input features: {input_features}")
        print(f"New model output features: {output_features}")
        # print(f"New model; {self.model}")
        # print("New Model:")
        # summary(self,input_size=(1,32,32))
