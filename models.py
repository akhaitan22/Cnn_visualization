import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import operations as operation
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
STRIDE = 1
AFFINE = False 

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def load_data():
	transform = transforms.Compose(
	    [transforms.ToTensor(),
     	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

	batch_size = 4

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=2)
	testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
	testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
	return trainloader, testloader

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

class DARTS(nn.Module):
    def __init__(self, num_channels=8, n_nodes=5):
        super(DARTS, self).__init__()
        self.ec00 = operation.NUM_OPS[2](C = 3, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.ec01 = operation.NUM_OPS[2](C = 3, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.ec10 = operation.NUM_OPS[2](C = 3, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.ec11 = operation.NUM_OPS[2](C = 3, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.ec12 = operation.NUM_OPS[2](C = 3, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e03 = operation.NUM_OPS[5](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e04 = operation.NUM_OPS[2](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e14 = operation.NUM_OPS[2](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e24 = operation.NUM_OPS[2](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e34 = operation.NUM_OPS[2](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.fc1 = nn.Linear(40960, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, n_nodes=5):
        ec00 = self.ec00(x)
        ec01 = self.ec01(x)
        ec10 = self.ec10(x)
        ec11 = self.ec11(x)
        ec12 = self.ec12(x)
        e0 = sum([ec00, ec10])
        e03 = self.e03(e0)
        e04 = self.e04(e0)
        e1 = sum([ec01, ec11])
        e14 = self.e14(e1)
        e24 = self.e24(ec12)
        e34 = self.e34(e03)
        e4 = torch.cat((e04, e14, e24, e34), dim=1)
        #print(e4.shape)
        x = torch.flatten(e4, 1)
        #print(x.shape)
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc2(x)
        #print(x.shape)
        return x

class Level1Op0(nn.Module):
    def __init__(self, channels_in):
        super(Level1Op0, self).__init__()
        if channels_in == 3:
            c=3
        else:
            c=10
        self.e01 = operation.NUM_OPS[3](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e02 = operation.NUM_OPS[2](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        #self.e03 = operation.NUM_OPS[7](C = 3, stride = STRIDE, affine = AFFINE)
        self.e04 = operation.NUM_OPS[1](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e12 = operation.NUM_OPS[7](C = 10, stride = STRIDE, affine = AFFINE)
        self.e13 = operation.NUM_OPS[8](C = 10, stride = STRIDE, affine = AFFINE)
        self.e14 = operation.NUM_OPS[7](C = 10, stride = STRIDE, affine = AFFINE)
        self.e23 = operation.NUM_OPS[3](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e24 = operation.NUM_OPS[5](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e34 = operation.NUM_OPS[7](C = 10, stride = STRIDE, affine = AFFINE)

    def forward(self, x):
    	e01 = self.e01(x)
    	e02 = self.e02(x)
    	#e03 = self.e03(x)
    	e04 = self.e04(x)
    	e12 = self.e12(e01)
    	e13 = self.e13(e01)
    	e14 = self.e14(e01)
    	e2 = sum([e02, e12])
    	e23 = self.e23(e2)
    	e24 = self.e24(e2)
    	#e3 = sum([e03, e13, e23])
    	e3 = sum([e13, e23])
    	e34 = self.e34(e3)
    	#return sum([e04, e14, e24, e34])
    	return sum([e14, e24, e34])

class Level1Op1(nn.Module):
    def __init__(self, channels_in):
        super(Level1Op1, self).__init__()
        if channels_in == 3:
            c=3
        else:
            c=10
        self.e01 = operation.NUM_OPS[2](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        #self.e02 = operation.NUM_OPS[8](C = c, stride = STRIDE, affine = AFFINE)
        #self.e03 = operation.NUM_OPS[7](C = c, stride = STRIDE, affine = AFFINE)
        self.e04 = operation.NUM_OPS[4](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        #self.e12 = operation.NUM_OPS[7](C = c, stride = STRIDE, affine = AFFINE)
        self.e13 = operation.NUM_OPS[5](C = 10, C2 = 10, stride = STRIDE, affine=AFFINE)
        self.e14 = operation.NUM_OPS[5](C = 10, C2 = 10, stride = STRIDE, affine=AFFINE)
        self.e23 = operation.NUM_OPS[2](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e24 = operation.NUM_OPS[6](C = c, C2 = 10, stride = STRIDE, affine = AFFINE)
        self.e34 = operation.NUM_OPS[4](C = 10, C2 = 10, stride = STRIDE, affine = AFFINE)

    def forward(self, x):
    	e01 = self.e01(x)
    	e02 = self.e02(x)
    	#e03 = self.e03(x)
    	e04 = self.e04(x)
    	e12 = self.e12(e01)
    	e13 = self.e13(e01)
    	e14 = self.e14(e01)
    	e2 = sum([e02, e12])
    	e23 = self.e23(e2)
    	e24 = self.e24(e2)
    	e3 = sum([e13, e23])
    	e34 = self.e34(e3)
    	return sum([e04, e14, e24, e34])

class HDARTS(nn.Module):
    def __init__(self, channels_in = 3, n_nodes=5):
        super(HDARTS, self).__init__()
        self.e02 = Level1Op1(3)
        self.e12 = Level1Op1(3)
        self.e23 = Level1Op1(10)
        self.fc1 = nn.Linear(10240, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, n_nodes=5):
    	print(x.shape)
    	e02 = self.e02(x)
    	e12 = self.e12(x)
    	e2 = sum([e02, e12])
    	e23 = self.e23(e2)
    	print(x.shape)
    	x = torch.flatten(e2, 1)
    	print(x.shape)
    	x = self.fc1(x)
    	print(x.shape)
    	x = self.fc2(x)
    	print(x.shape)
    	return x

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(
        nn.Conv2d(3, 6, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 5),
        nn.ReLU(),
        nn.MaxPool2d(2, 2))
        self.next = nn.Sequential(
        nn.Linear(16 * 5 * 5, 120),
        nn.Linear(120, 84),
        nn.Linear(84, 10))
        
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.next(x)
        return x
 

def train(model, trainloader, optimizer, epochs = 1, criterion = nn.CrossEntropyLoss()):
	for epoch in range(epochs):  # loop over the dataset multiple times
		running_loss = 0.0
		for i, data in enumerate(trainloader, 0):
			if(i>200):
				break
			inputs, labels = data
			optimizer.zero_grad()
			outputs = model(inputs)
			#print(outputs.shape, labels.shape)
			#print(outputs, labels)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
			if (i+1) % 100 == 0:    # print every 2000 mini-batches
				print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 100))
				#print(outputs.shape, labels.shape)
				#print(outputs, labels)
				running_loss = 0.0

def test(model, testloader):
	correct = 0
	total = 0
	model.eval()
	with torch.no_grad():
		for data in testloader:
			images, labels = data
			outputs = model(images)
			if(total>4):
				break;
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
	PATH = './darts.pth'
	model = HDARTS()
	trainloader, testloader = load_data()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	train(model, trainloader, optimizer)
	for data in testloader:
			images, labels = data
			if(True):
				break
	#print(images[0,:], labels[0])
	#model.load_state_dict(torch.load(PATH))
	print('Finished Training')
	#torch.save(model.state_dict(), PATH)
	test(model, testloader)
	#print(operations.OPS['avg_pool_3x3'](C=3, C2=10, stride=STRIDE, affine=AFFINE))
