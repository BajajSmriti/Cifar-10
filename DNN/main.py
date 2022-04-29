from operator import itemgetter
import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from Cifar_DNN import CifarDNN
import matplotlib.pyplot as plt

ROOT_PATH='D:\\Northeastern_University\\SML\\cifar-10-python\\cifar-10-batches-py\\'

BATCH_SIZE=100

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = CIFAR10(root=ROOT_PATH, download=True, train=True, 
    transform=transform)
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, [40000, 10000])
eval_dataset = CIFAR10(root=ROOT_PATH, train=False, 
    transform=transform)

train_data_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_data_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
eval_data_loader = DataLoader(dataset=eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
inputSize = 32 * 32 * 3
outputSize = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device:', device)
model = CifarDNN(inputSize, outputSize).to(device)
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
            for data, target in valid_data_loader:
                data = data.to(device)
                target = target.to(device)
                output = model(data)
                criterion = nn.CrossEntropyLoss()
                test_loss = criterion(output, target)
                pred = output.data.max(1, keepdim = True)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum()

            perValue = correct / float(len(eval_data_loader.dataset))
            history.append({"loss_value": loss.cpu().detach().numpy(),"test_loss": test_loss.cpu().detach().numpy(), "accuracy": perValue.cpu().detach().numpy(), "epoch": epoch, "step": step})
            test_loss /= len(eval_data_loader.dataset)
            print(
                'Test set: Epoch [{}]:Step[{}] Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
                .format(
                    epoch, step, test_loss, correct, len(eval_data_loader.dataset),
                    float(100. * perValue)
                )
            )    

torch.save(model.state_dict(), "./dnnModel.mod")

model.load_state_dict(torch.load("./dnnModel.mod"))

legend = ['Train', 'Validation']
plt.plot(list(map(itemgetter("loss_value"), history)))
plt.plot(list(map(itemgetter("test_loss"), history)))
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(legend, loc='upper left')
plt.show()

legend = ['Validation']
plt.plot(list(map(itemgetter("accuracy"), history)))
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(legend, loc='upper left')
plt.show()

test_loss = 0
correct = 0
for data, target in eval_data_loader:
    data = data.to(device)
    target = target.to(device)
    output = model(data)
    criterion = nn.CrossEntropyLoss()
    test_loss = criterion(output, target)
    pred = output.data.max(1, keepdim = True)[1]
    correct += pred.eq(target.data.view_as(pred)).cpu().sum()

perValue = correct / float(len(eval_data_loader.dataset))
history.append({"loss_value": loss, "accuracy": perValue})
test_loss /= len(eval_data_loader.dataset)
print(
    'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'
    .format(
        test_loss, correct, len(eval_data_loader.dataset),
        float(100. * perValue)
    )
)    
