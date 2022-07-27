import torch
import torch.nn as nn
# from torchsummary import summary
import copy


class initialization_class():
  kaiming_uniform = 'kaiming_uniform' # by default fan_in mode
  kaiming_normal = 'kaiming_normal'
  kaiming_uniform_fan_out = 'kaiming_uniform_fan_out'
  kaiming_normal_fan_out = 'kaiming_normal_fan_out'
  xavier_uniform = 'xavier_uniform'
  xavier_normal = 'xavier_normal'
  random = 'random'
  zeros = 'zeros'
  negative = 'negative'


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
      self.dropout = nn.Dropout(0.25)
      self.flatten = nn.Flatten()
      self.linear = nn.Linear(128 * 3 * 3, out_features=10)

  def forward(self, input_data):
    x = self.conv1(input_data)
    x= self.dropout(x)
    x = self.conv2(x)
    x= self.dropout(x)
    x = self.conv3(x)
    x= self.dropout(x)
    x = self.conv4(x)
    x = self.dropout(x)
    x = self.flatten(x)
    logits = self.linear(x)
    return logits

  def freeze_old_params(self):
    #freeze all layers except last liner layer
    for names, param in self.named_parameters():
      if 'linear' not in names:
          param.requires_grad = False
      # print(f'name: {names}, {param.requires_grad}')
  
  def unfreeze_old_params(self):
    #unfreeze all layers except last liner layer
    for param in self.parameters():
        param.requires_grad = True

  def addNovelClassesToModel(self, noNovelClasses, initialization = initialization_class.random):
      #add new novel classes to pre-trained model
      #add new class parameters , here we have added 3 new classes
      input_features = self.linear.in_features
      output_features = self.linear.out_features
      
      print("Old Model:")
      # summary(self,input_size=(1,32,32))

      #save fc3 layers weights for initialization step
      old_weights = copy.deepcopy(self.linear.weight.data)  #torch.Size([10, 256])
      old_bias = copy.deepcopy(self.linear.bias.data)  #torch.Size([10, 256])

      new_output_features = output_features + noNovelClasses
      print(f"new output features: {new_output_features}")

      self.linear = nn.Linear(in_features=input_features, out_features= new_output_features)

      #Initialize new weights with kaiming normal distribution 
      # in paper xaviour is suggested for relu kaiming is recommended
      if initialization == initialization_class.kaiming_uniform:
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
      elif initialization == initialization_class.kaiming_normal:
        nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
      elif initialization == initialization_class.kaiming_uniform_fan_out:
        nn.init.kaiming_uniform_(self.linear.weight, mode='fan_out', nonlinearity='relu')
      elif initialization == initialization_class.kaiming_normal_fan_out:
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
      elif initialization == initialization_class.xavier_uniform:
        nn.init.xavier_uniform_(self.linear.weight)
      elif initialization == initialization_class.xavier_normal:
        nn.init.xavier_normal_(self.linear.weight)
      elif initialization == initialization_class.xavier_normal:
        nn.init.xavier_normal_(self.linear.weight)
      elif initialization == initialization_class.zeros:
        self.linear.weight.data.fill_(0)
      elif initialization == initialization_class.negative:
        self.linear.weight.data.fill_(-1)
      #copy old weights for old paramerter
      self.linear.weight.data[:output_features] = old_weights
      self.linear.bias.data[:output_features] = old_bias

      input_features = self.linear.in_features
      output_features = self.linear.out_features
      print(f"New model input features: {input_features}")
      print(f"New model output features: {output_features}")
      # print(f"New model; {self.model}")
      print("New Model:")
      # summary(self,input_size=(1,32,32))
      return