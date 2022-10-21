from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torch.autograd import Variable
import matplotlib.pyplot as plt


class L_SVM(nn.Module):
    def __init__(self):
        super(L_SVM, self).__init__()

        self.layer = nn.Linear(28 * 28, 1)

    def forward(self, x):

        output = self.layer(x)

        return output

class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()

        self.layer = nn.Linear(28 * 28, 1)

    def forward(self, x):

        outputs = self.layer(x)

        return outputs

def hinge_loss_LSVM(outputs, labels):

    loss = torch.mean(torch.maximum(torch.zeros_like(labels).squeeze(), 1 - labels.squeeze() * (outputs.squeeze())))

    return loss

def Loss_LGR(output, label):

    #size of output and label is both (64,1)
    loss = torch.mean(torch.log(1 + torch.exp((-1)*label.squeeze() * (output.squeeze()))))
    
    return loss


def train(args, model, device, train_loader, optimizer, epoch):

    model.train()

    total_loss_this_epoch = torch.zeros(500)

    for batch_idx, (data, label) in enumerate(train_loader):
        data = Variable(data.view(-1, 28*28))
        label = Variable(2*(label.float()-0.5))

        data, label = data.to(device), label.to(device) #size of data: (64, 1, 28, 28)
        optimizer.zero_grad()
        output = model(data)
        loss = hinge_loss_LSVM(output, label)

        total_loss_this_epoch[batch_idx] = loss

        loss.backward()
        optimizer.step()

        #print the result during training
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    average_loss_this_epoch = total_loss_this_epoch[:batch_idx+1].mean()

    return average_loss_this_epoch


def test(model, device, test_loader):

    model.eval()
    correct = 0.
    total = 0.
    with torch.no_grad():
        for data, target in test_loader:

            data = Variable(data.view(-1, 28*28))

            data, target = data.to(device), target.to(device)
            output = model(data)

            correct += (output.view(-1).heaviside(torch.tensor([1.]).to(device)) == target).sum()
            total += data.shape[0]
            print('Accuracy of the model on the test images: %f %%' % (100 * (correct.float() / total)))

    return 100 * (correct.float() / total)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example of Machine Learning Course')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: 14)')
                        
    parser.add_argument('--lr', type=float, default=10, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--optimizer', default="SGD",
                        help='The optimizer used in this model, SGD or SGD-M')
    parser.add_argument('--SGD_momentum', default=0.9)
    parser.add_argument('--model', default="L_SVM",
                        help='The model used, L_SVM or Logistic')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--seed', type=int, default=1000, metavar='S',
                        help='random seed (default: 10)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    
    #load the training dataset
    train_data = datasets.MNIST('/home/yuzhi/ML_HW_2/data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))

    #load the test dataset
    test_data = datasets.MNIST('/home/yuzhi/ML_HW_2/data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
        
    subset_indices = ((train_data.train_labels == 0) + (train_data.train_labels == 1)).nonzero().reshape(-1)
    train_loader = torch.utils.data.DataLoader(train_data,batch_size=args.batch_size, shuffle=False, sampler=SubsetRandomSampler(subset_indices))

    subset_indices = ((test_data.test_labels == 0) + (test_data.test_labels == 1)).nonzero().reshape(-1)
    test_loader = torch.utils.data.DataLoader(test_data,batch_size=args.batch_size, shuffle=False,sampler=SubsetRandomSampler(subset_indices))

    if args.model == "Logistic":
        model = LogisticRegression().to(device)
    elif args.model == "L_SVM":
        model = L_SVM().to(device)
    else:
        print("wrong model input information")

    if args.optimizer == "SGD":    
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == "SGD-M":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.SGD_momentum)
    else:
        print("wrong optimizer input information")

    train_loss_all_epochs = torch.zeros(args.epochs)
    for epoch in range(1, args.epochs + 1):
        average_loss_this_epoch = train(args, model, device, train_loader, optimizer, epoch)
        train_loss_all_epochs[epoch-1] = average_loss_this_epoch

    plt.plot(train_loss_all_epochs.detach().numpy())
    plt.xlabel("Epoch Number")
    plt.ylabel("Train Loss")
    plt.savefig("train_loss_LR{}_{}_{}.png".format(args.lr, args.optimizer, args.model))

    torch.save(train_loss_all_epochs, "train_loss_all_epochs_LR{}_{}_{}.pt".format(args.lr, args.optimizer, args.model))

    test_accuracy = test(model, device, test_loader)

    with open("Test_accuracy_LR{}_{}_{}.txt".format(args.lr, args.optimizer, args.model), 'w') as f:
        f.write(str(test_accuracy.detach().cpu().numpy()))
        f.write('\n')


if __name__ == '__main__':
    main()


