import torch
import torch.nn as nn

class VGGNet(nn.Module):

  def __init__(self):
      super().__init__()

      self.conv1 = nn.Sequential(
          nn.Conv2d(
              in_channels=1,
              out_channels=16,
              kernel_size=3,
              stride=1,
              padding=2
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)
      )

      self.conv2 = nn.Sequential(
          nn.Conv2d(
              in_channels=16,
              out_channels=32,
              kernel_size=3,
              stride=1,
              padding=2
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)
      )

      self.conv3 = nn.Sequential(
          nn.Conv2d(
              in_channels=32,
              out_channels=64,
              kernel_size=3,
              stride=1,
              padding=2
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)
      )

      self.conv4 = nn.Sequential(
          nn.Conv2d(
              in_channels=64,
              out_channels=128,
              kernel_size=3,
              stride=1,
              padding=2
          ),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2)
      )
      self.flatten = nn.Flatten()
      self.linear = nn.Linear(128 * 3 * 3, out_features=10)

  def forward(self, input_data):
    x = self.conv1(input_data)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.conv4(x)
    x = self.flatten(x)
    logits = self.linear(x)
    return logits

  def addNovelClassesToModel(self, noNovelClasses):
      #add new novel classes to pre-trained model
      #add new class parameters , here we have added 3 new classes
      input_features = self.linear.in_features
      output_features = self.linear.out_features
      
      print("Old Model:")
      summary(model,input_size=(1,32,32))

      #save fc3 layers weights for initialization step
      old_weights = copy.deepcopy(self.linear.weight.data)  #torch.Size([10, 256])

      new_output_features = output_features + noNovelClasses
      print(f"new output features: {new_output_features}")

      self.linear = nn.Linear(in_features=input_features, out_features= new_output_features)

      #Initialize new weights with kaiming normal distribution 
      # in paper xaviour is suggested for relu kaiming is recommended
      nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
      #copy old weights for old paramerter
      self.linear.weight.data[:output_features] = old_weights

      input_features = self.linear.in_features
      output_features = self.linear.out_features
      print(f"New model input features: {input_features}")
      print(f"New model output features: {output_features}")
      # print(f"New model; {self.model}")
      print("New Model:")
      summary(self,input_size=(1,32,32))

      return