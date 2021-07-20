import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


#우리가 가진 MNIST 데이터는 (1784)의 데이터 형태를 가짐
#구분하려는 숫자의 종류가 총 10가지임

#CNN 구성 (Define Neural Networks Nodel)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1) #행벡터로 고치지 않고 2D 그대로 연산함. -> Linear는 입력 및 출력을 정해줘서 벡터의 모양을 바꿔줌.
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
print("init model done")


#Set Hyper Parameters and other variables to train the model.

batch_size = 64
test_batch_size = 1000
epochs = 100
lr = 0.01
momentum = 0.5
no_cuda = True
seed = 1
log_interval = 200

use_cuda = not no_cuda and torch.cuda.is_available()

torch.manual_seed(seed)

device = torch.device("cuda" if use_cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

print("set vars and device done")




#Prepare Data Loader for Training and Validation

transform = transforms.Compose(([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081))
]))

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True, transform = transform),
    batch_size = batch_size, shuffle=True, **kwargs
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True, transform=transform),
    batch_size=test_batch_size, shuffle=True, **kwargs

    
)

#Optimizer

model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)



#Define Train function and Test function to validate.

def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device) # send data to gpu or cpu
        optimizer.zero_grad() # set gradient zero.
        output = model(data) # get output from model
        loss = F.nll_loss(output, target) # calculate nll loss
        loss.backward() #do backpropagation
        optimizer.step() # update weight and biases
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx*len(data), len(train_loader.dataset), 100.*batch_idx/len(train_loader), loss.item()))




def test(log_interval, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format
          (test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# Train and Test the model and save it.

for epoch in range(1, 11):
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)
    torch.save({'model' : model.state_dict, 'optimizer' : optimizer.state_dict, 'epoch': epoch}, './model.pt')


