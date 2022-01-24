import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

#Below methods to claculate input featurs to the FC layer
#and weight initialization for CNN model is based on the below github repo
#Based on :https://github.com/Lab41/cyphercat/blob/master/Utils/models.py
 
def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out
    
    
def size_max_pool(size, kernel, stride=None, padding=0): 
    if stride == None: 
        stride = kernel
    out = int(((size - kernel + 2*padding)/stride) + 1)
    return out

#Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_cifar(size):
    feat = size_conv(size,3,1,1)
    feat = size_max_pool(feat,2,2)
    feat = size_conv(feat,3,1,1)
    out = size_max_pool(feat,2,2)
    return out
    
#Calculate in_features for FC layer in Shadow Net
def calc_feat_linear_mnist(size):
    feat = size_conv(size,5,1)
    feat = size_max_pool(feat,2,2)
    feat = size_conv(feat,5,1)
    out = size_max_pool(feat,2,2)
    return out

#Parameter Initialization
def init_params(m): 
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear): 
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)

#####################################################
# Define Target, Shadow and Attack Model Architecture
#####################################################

#Target Model
class TargetNet(nn.Module):
    def __init__(self, input_dim, hidden_layers, size, out_classes):
        super(TargetNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out
        
    
#Shadow Model mimicking target model architecture, for our implememtation is different than target
class ShadowNet(nn.Module):
    def __init__(self, input_dim, hidden_layers,size,out_classes):
        super(ShadowNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=hidden_layers[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[0]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_layers[0], out_channels=hidden_layers[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers[1]),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        features = calc_feat_linear_cifar(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((features**2 * hidden_layers[1]), hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layers[2], out_classes)
        )
        
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

#Pretrained VGG11 model for Target
class VggModel(nn.Module):
    def __init__(self, num_classes,layer_config,pretrained=True):
        super(VggModel, self).__init__()
        #Load the pretrained VGG11_BN model
        if pretrained:
            pt_vgg = models.vgg11_bn(pretrained=pretrained)

            #Deleting old FC layers from pretrained VGG model
            print('### Deleting Avg pooling and FC Layers ####')
            del pt_vgg.avgpool
            del pt_vgg.classifier

            self.model_features = nn.Sequential(*list(pt_vgg.features.children()))
            
            #Adding new FC layers with BN and RELU for CIFAR10 classification
            self.model_classifier = nn.Sequential(
                nn.Linear(layer_config[0], layer_config[1]),
                nn.BatchNorm1d(layer_config[1]),
                nn.ReLU(inplace=True),
                nn.Linear(layer_config[1], num_classes),
            )

    def forward(self, x):
        x = self.model_features(x)
        x = x.squeeze()
        out = self.model_classifier(x)
        return out

#Target/Shadow Model for MNIST
class MNISTNet(nn.Module):
    def __init__(self, input_dim, n_hidden,out_classes=10,size=28):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_dim, out_channels=n_hidden, kernel_size=5),
            nn.BatchNorm2d(n_hidden),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=n_hidden, out_channels=n_hidden*2, kernel_size=5),
            nn.BatchNorm2d(n_hidden*2),
            # nn.Dropout(p=0.5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        features = calc_feat_linear_mnist(size)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(features**2 * (n_hidden*2), n_hidden*2),
            nn.ReLU(inplace=True),
            nn.Linear(n_hidden*2, out_classes)
        )
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.classifier(out)
        return out

#Attack MLP Model
class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64,out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )    
    def forward(self, x):
        out = self.classifier(x)
        return out
    
class MLP1(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP1, self).__init__()
        print("NN: MLP is created")
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        # self.softmax = nn.Softmax(dim=1)

        # weights_init = 0.001
        # bias_init = 0.001
        #
        # nn.init.constant_(self.layer_input.weight,weights_init)
        # nn.init.constant_(self.layer_input.bias, bias_init)
        # nn.init.constant_(self.layer_hidden.weight, weights_init)
        # nn.init.constant_(self.layer_hidden.bias, bias_init)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        # x = x.view(-1, 1)
        x = self.layer_input(x)
        x = self.layer_hidden(x)
        return F.log_softmax(x, dim=1)
    
class CNNMnist(nn.Module):
    def __init__(self, num_classes):
        super(CNNMnist, self).__init__()
        print("NN: CNNMnist is created")
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    
class CNNCifar(nn.Module):
    def __init__(self, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)        
        self.fc1 = nn.Linear(256 * 2 * 2, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(-1, 256 * 2 * 2)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
