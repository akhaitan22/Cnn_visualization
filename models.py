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
        self.e12 = conv(3, 10)
        self.e13 = conv(3, 10)
        self.e14 = conv(3, 10)
        self.e15 = conv(3, 10)
        self.e23 = conv(10, 10)
        self.e24 = conv(10, 10)
        self.e25 = conv(10, 10)
        self.e34 = conv(10, 10)
        self.e35 = conv(10, 10)
        self.e45 = conv(10, 10)
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, n_nodes=5):
        e12 = self.e12(x)
        e13 = self.e13(x)
        e14 = self.e14(x)
        e15 = self.e15(x)
        e23 = self.e23(e12)
        e24 = self.e24(e12)
        e25 = self.e25(e12)
        e3 = sum([e13, e23])
        e34 = self.e34(e3)
        e35 = self.e35(e3)
        e4 = sum([e14, e24, e34])
        e45 = self.e45(e4)
        e5 = torch.cat((e15, e25, e35, e45), dim=1)
        #print(e5.shape)
        x = torch.flatten(x, 1)
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
            channels_2 = 10
        else:
            channels_2 = channels_in
        self.e01 = operation.NUM_OPS[3](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e02 = operation.NUM_OPS[2](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e03 = operation.NUM_OPS[7](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e04 = operation.NUM_OPS[1](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e12 = operation.NUM_OPS[7](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e13 = operation.NUM_OPS[8](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e14 = operation.NUM_OPS[7](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e23 = operation.NUM_OPS[3](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e24 = operation.NUM_OPS[5](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e34 = operation.NUM_OPS[7](C = channels_in, stride = STRIDE, affine = AFFINE)

    def forward(self, x):
    	e01 = self.e01(x)
    	e02 = self.e02(x)
    	e03 = self.e03(x)
    	e04 = self.e04(x)
    	e12 = self.e12(e01)
    	e13 = self.e13(e01)
    	e14 = self.e14(e01)
    	e2 = sum([e02, e12])
    	e23 = self.e23(e2)
    	e24 = self.e24(e2)
    	e3 = sum([e03, e13, e23])
    	e34 = self.e34(e3)
    	return sum([e04, e14, e24, e34])

class Level1Op1(nn.Module):
    def __init__(self, channels_in):
        super(Level1Op1, self).__init__()
        if channels_in == 3:
            channels_2 = 10
        else:
            channels_2 = channels_in
        self.e01 = operation.NUM_OPS[2](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e02 = operation.NUM_OPS[8](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e03 = operation.NUM_OPS[7](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e04 = operation.NUM_OPS[4](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e12 = operation.NUM_OPS[7](C = channels_in, stride = STRIDE, affine = AFFINE)
        self.e13 = operation.NUM_OPS[5](C = channels_in, C2 = channels_2, stride = STRIDE, affine=AFFINE)
        self.e14 = operation.NUM_OPS[5](C = channels_in, C2 = channels_2, stride = STRIDE, affine=AFFINE)
        self.e23 = operation.NUM_OPS[2](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e24 = operation.NUM_OPS[6](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)
        self.e34 = operation.NUM_OPS[4](C = channels_in, C2 = channels_2, stride = STRIDE, affine = AFFINE)

    def forward(self, x):
    	e01 = self.e01(x)
    	e02 = self.e02(x)
    	e03 = self.e03(x)
    	e04 = self.e04(x)
    	e12 = self.e12(e01)
    	e13 = self.e13(e01)
    	e14 = self.e14(e01)
    	e2 = sum([e02, e12])
    	e23 = self.e23(e2)
    	e24 = self.e24(e2)
    	e3 = sum([e03, e13, e23])
    	e34 = self.e34(e3)
    	return sum([e04, e14, e24, e34])

class HDARTS(nn.Module):
    def __init__(self, channels_in = 3, n_nodes=5):
        super(HDARTS, self).__init__()
        self.e01 = Level1Op0(3)
        self.e02 = Level1Op0(3)
        self.e12 = Level1Op1(10)
        self.fc1 = nn.Linear(3072, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x, n_nodes=5):
    	print(x.shape)
    	e01 = self.e01(x)
    	e02 = self.e02(x)
    	e12 = self.e23(e01)
    	print(x.shape)
    	x = torch.flatten(x, 1)
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
			ig = IntegratedGradients(model)
			baseline=torch.zeros(images.shape)
			if(total>4):
				break;
			attributions, delta = ig.attribute(images, labels, target=0, return_convergence_delta=True)
			print('IG Attributions:', attributions)
			print('Convergence Delta:', delta)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == '__main__':
	PATH = './net.pth'
	model = Net2()
	trainloader, testloader = load_data()
	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	train(model, trainloader, optimizer)
	#model.load_state_dict(torch.load(PATH))
	print('Finished Training')
	#torch.save(model.state_dict(), PATH)
	test(model, testloader)
	#print(operations.OPS['avg_pool_3x3'](C=3, C2=10, stride=STRIDE, affine=AFFINE))
