from pickletools import optimize
import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Cifar_DNN import CifarDNN

ROOT_PATH='D:\\Northeastern_University\\SML\\cifar-10-python\\cifar-10-batches-py\\'

BATCH_SIZE=100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = CIFAR10(root=ROOT_PATH, download=True, train=True, 
    transform=transform)
eval_dataset = CIFAR10(root=ROOT_PATH, train=False, 
    transform=transform)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
inputSize = 32 * 32 * 3
outputSize = 10
model = CifarDNN(inputSize, outputSize).cuda()
optimizer = torch.optim.SGD(model.parameters(), momentum=0.7, lr=0.01)
loss_func = torch.nn.CrossEntropyLoss()
history = []
for epoch in range(20):
    for step, (input, target) in enumerate(train_data_loader):
        input = input.cuda()
        target = target.cuda()
        model.train()
        output = model(input)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            model.eval()
            test_loss = 0
            correct = 0
            for data, target in eval_data_loader:
                data = data.cuda()
                target = target.cuda()
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                test_loss = criterion(output, target)
                pred = output.data.max(1, keepdim = True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            perValue = correct / float(len(eval_data_loader.dataset))
            history.append({"loss_value": loss, "accuracy": perValue, "epoch": epoch, "step": step})
            test_loss /= len(eval_data_loader.dataset)
            print(
                'Test set: Epoch [{}]:Step[{}] Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
                .format(
                    epoch, step, test_loss, correct, len(eval_data_loader.dataset),
                    float(100. * perValue)
                )
            )    

torch.save(model.state_dict(), "D:\\Northeastern_University\\SML\\cifar-10-python\\model")


