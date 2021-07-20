import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import os.path
import copy

#우리가 가진 MNIST 데이터는 (1784)의 데이터 형태를 가짐
#구분하려는 숫자의 종류가 총 10가지임
'''
1. 모델 저장 및 불러오는 함수 만들기 -> train 함수 안에서 이제 위에서 저장해놓은거 불러오고, 끝에서 저장하게! => 구현 완료.

지금 구현이 안 된 것 : 
2. best accuracy 저장하기
3. 그리고 이걸 서버컴퓨터의 GPU를 이용해서 돌려보기!!!
'''




#DNN 구성 (Define Neural Networks Nodel)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.float()
        h1 = F.relu(self.fc1(x.view(-1, 784)))
        h2 = F.relu(self.fc2(h1))
        h3 = F.relu(self.fc3(h2))
        h4 = F.relu(self.fc4(h3))
        h5 = F.relu(self.fc5(h4))
        h6 = self.fc6(h5)
        return F.log_softmax(h6, dim=1)
    

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



#Define Train function and Test function to validate.====================================

def train(log_interval, model, device, train_loader, optimizer, epoch):
    
    
    file = './model_saved.pt'
    if os.path.isfile(file):
        #model = torch.load('./model_saved.tar')
        
        checkpoint = torch.load('./model_saved.pt')
        #checkpoint.train()
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        epoch = epoch + 1 #여기 주의. 이미 저장된 부분(에포크)의 다음 에포크부터 돼야 하므로 이렇게 기록 및 불러옴.




    # checkpoint = torch.load('./model_saved.tar')
    # model.load_state_dict(checkpoint['model'])


    model.train()
    
#     model = torch.load('./model_saved.pt') #전체 모델을 통째로 불러옴. 클래스 선언 필수.
#     #model.load_state_dict(torch.load('./model_saved.pt')) #state_dict를 불러 온 후, 모델에 저장.
#     checkpoint = torch.load(PATH + 'all.tar')   # dict 불러오기

# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])




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






# Train and Test the model and save it.===============================================

for epoch in range(1, epochs):
    #model = TheModelClass(*args, **kwargs)
    #model.load_state_dict(torch.load(PATH))
    #model.eval() -> 이건 train함수랑 test함수 안에서 알아서 다 잘 해 줌.
    train(log_interval, model, device, train_loader, optimizer, epoch)
    test(log_interval, model, device, test_loader)

    torch.save({'model' : model.state_dict(), 'optimizer' : optimizer.state_dict(), 'epoch': epoch }, './model_saved.pt') #가장 마지막이 아니라 train 및 test가 끝난 매 에포크마다 저장.


#torch.save({'model' : model.state_dict, 'optimizer' : optimizer.state_dict, 'epoch': epoch }, './model_saved.pt')
    
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#model.load_state_dict(torch.load("model_state.pth", device))
