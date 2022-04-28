import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import PIL

def run():
    train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomAffine(degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.9, 1.1), resample=PIL.Image.BILINEAR),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 128

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=train_transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000])

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                            shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # functions to show an image


    def imshow(img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


    # get some random training images
    dataiter = iter(trainloader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 128, 5, padding=2)
            self.conv2 = nn.Conv2d(128, 128, 5, padding=2)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.conv4 = nn.Conv2d(256, 256, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.bn_conv1 = nn.BatchNorm2d(128)
            self.bn_conv2 = nn.BatchNorm2d(128)
            self.bn_conv3 = nn.BatchNorm2d(256)
            self.bn_conv4 = nn.BatchNorm2d(256)
            self.bn_dense1 = nn.BatchNorm1d(1024)
            self.bn_dense2 = nn.BatchNorm1d(512)
            self.dropout_conv = nn.Dropout2d(p=0.25)
            self.dropout = nn.Dropout(p=0.5)
            self.fc1 = nn.Linear(256 * 8 * 8, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 10)

        def conv_layers(self, x):
            out = F.relu(self.bn_conv1(self.conv1(x)))
            out = F.relu(self.bn_conv2(self.conv2(out)))
            out = self.pool(out)
            out = self.dropout_conv(out)
            out = F.relu(self.bn_conv3(self.conv3(out)))
            out = F.relu(self.bn_conv4(self.conv4(out)))
            out = self.pool(out)
            out = self.dropout_conv(out)
            return out

        def dense_layers(self, x):
            out = F.relu(self.bn_dense1(self.fc1(x)))
            out = self.dropout(out)
            out = F.relu(self.bn_dense2(self.fc2(out)))
            out = self.dropout(out)
            out = self.fc3(out)
            return out

        def forward(self, x):
            out = self.conv_layers(x)
            out = out.view(-1, 256 * 8 * 8)
            out = self.dense_layers(out)
            return out

    net = Net()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Number of trainable parameters:", num_params)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, min_lr=0)

    loss_hist, acc_hist = [], []
    loss_hist_val, acc_hist_val = [], []

    for epoch in range(10):
        running_loss = 0.0
        correct = 0
        for data in trainloader:
            batch, labels = data
            batch, labels = batch.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # compute training statistics
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_set)
        avg_acc = correct / len(train_set)
        loss_hist.append(avg_loss)
        acc_hist.append(avg_acc)

        # validation statistics
        net.eval()
        with torch.no_grad():
            loss_val = 0.0
            correct_val = 0
            for data in val_loader:
                batch, labels = data
                batch, labels = batch.to(device), labels.to(device)

                outputs = net(batch)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                loss_val += loss.item()
            avg_loss_val = loss_val / len(val_set)
            avg_acc_val = correct_val / len(val_set)
            loss_hist_val.append(avg_loss_val)
            acc_hist_val.append(avg_acc_val)
        net.train()

        scheduler.step(avg_loss_val)
        print('[epoch %d] loss: %.5f accuracy: %.4f val loss: %.5f val accuracy: %.4f' % (epoch + 1, avg_loss, avg_acc, avg_loss_val, avg_acc_val))

    print('Finished Training')

    legend = ['Train', 'Validation']
    plt.plot(loss_hist)
    plt.plot(loss_hist_val)
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(legend, loc='upper left')
    plt.show()

    legend = ['Train', 'Validation']
    plt.plot(acc_hist)
    plt.plot(acc_hist_val)
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(legend, loc='upper left')
    plt.show()

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    dataiter = iter(testloader)
    images, labels = dataiter.next()

    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    pred_vec = []
    net.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            pred_vec.append(predicted)
        pred_vec = torch.cat(pred_vec)            

    print('Accuracy on the 10000 test images: %.2f %%' % (100 * correct / len(test_set)))


if __name__ == '__main__':
    run()