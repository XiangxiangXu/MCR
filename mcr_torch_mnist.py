
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 12
batch_size = 128
fdim = 128

num_classes = 10

tilde = lambda x: x - torch.mean(x, 0) # zero-mean
cov = lambda x: x.T @ x / (x.size()[0]-1.) # covariance

def neg_hscore(f, g):
    """
    compute the negative h-score
    """
    f0 = tilde(f) 
    g0 = tilde(g) 
    corr = torch.mean(torch.sum(f0*g0, 1))
    cov_f = cov(f0)
    cov_g = cov(g0)
    return - corr + torch.trace(cov_f @ cov_g) / 2.

def min_max_normalization(tensor, min_value, max_value):
    min_tensor = tensor.min()
    tensor = (tensor - min_tensor)
    max_tensor = tensor.max()
    tensor = tensor / max_tensor
    tensor = tensor * (max_value - min_value) + min_value
    return tensor


def tensor_round(tensor):
    return torch.round(tensor)


img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda tensor:min_max_normalization(tensor, 0, 1)),
    # transforms.Lambda(lambda tensor:tensor_round(tensor))
])

train_dataset = MNIST('../mnist_data/', transform=img_transform, download=True)
test_dataset = MNIST('../mnist_data/', train=False, transform=img_transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


class fNet(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc = nn.Linear(9216, fdim)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.dropout(x, p =0.25)
        f = x.flatten(start_dim = 1)
        f = self.fc(f)
        return f        #x.view(-1, x.size(1))                

class gNet(nn.Module):
    def __init__(self, fdim, num_classes):
        super().__init__()
        self.fc = nn.Linear(num_classes, fdim)

    def forward(self, y):
        g = self.fc(y)
        return g

class fgNet(nn.Module):
    def __init__(self, fdim, num_classes):
        super().__init__()
        self.fnet = fNet(fdim)
        self.num_classes = num_classes
        self.gnet = gNet(fdim, num_classes)
        
    def forward(self, x, y):
        f = self.fnet(x)
        g = self.gnet(y)
        return f, g

    # compute the prediction 
    def pred(self, x, py):
        f = self.fnet(x)
        f0 = tilde(f)
        g = self.gnet(torch.eye(self.num_classes))
        g0 = g - py @ g
        pygx = py * (1 + f0 @ g0.T)
        y_pred = torch.argmax(pygx, 1)
        return pygx, y_pred

fg_model = fgNet(fdim, num_classes)
opt = optim.Adadelta(fg_model.parameters())

# compute the prior distribution 
py = torch.mean(torch.eye(num_classes)[train_dataset.targets], 0)

for epoch in range(num_epochs):
    for x, y in train_loader:
        y_onehot = torch.eye(num_classes)[y] # one-hot encoding
        f, g = fg_model(x, y_onehot)
        loss = neg_hscore(f, g)

        loss.backward()
        opt.step()
        opt.zero_grad()

    print('training h_score =', neg_hscore(f, g))

# Test    
with torch.no_grad():
    for x, y in test_loader:
        y_onehot = torch.eye(num_classes)[y] # one-hot encoding
        f, g = fg_model(x, y_onehot)
        pygx, y_pred = fg_model.pred(x, py)
        acc = (y_pred == y).sum() / float(y.size(0))
        print('test_hscore =', neg_hscore(f, g), '| acc = ', acc)
