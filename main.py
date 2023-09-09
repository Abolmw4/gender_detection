import numpy as np
import torch.cuda
from torchvision.transforms import transforms
from data import AbolfazlDataset
from network import AbolfazNework
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt


def calculate_accuracy(y_pred, y, device):
    correct = (y_pred.argmax(1).to(device) == y.argmax(1).to(device)).type(torch.float).sum()
    acc = correct / y.shape[0]
    return acc


def train(model, iterator, optimizer, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.train()

    for (x, y) in tqdm(iterator, desc="Training", leave=True):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.float32)
        label = y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, label)
        acc = calculate_accuracy(y_pred, label, device)

        l2_lambda = 0.0008
        l2_norm = sum(p.pow(2.0).sum()
                      for p in model.parameters())

        loss = loss + l2_lambda * l2_norm

        loss.backward()
        optimizer.step()
        # acc = calculate_accuracy(y_pred, label, device)
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion, device):
    epoch_loss = 0.0
    epoch_acc = 0.0
    model.eval()

    for (x, y) in tqdm(iterator, desc="Evaluating", leave=True):
        x = x.to(device)
        y = torch.tensor(y, dtype=torch.float32)
        label = y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, label)
        acc = calculate_accuracy(y_pred, label, device)
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f'system device {device}')
    myNet = AbolfazNework().to(device)
    optimizer = optim.Adam(myNet.parameters(), lr=0.0001)
    # weight = torch.tensor((0.4914, 0.5086)).to(device)
    criterion = nn.BCELoss()
    number_of_epochs = 15
    batch_size = 32

    tr = transforms.Compose([transforms.ToTensor(),
                             transforms.Resize((224, 224)),
                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    data_train = AbolfazlDataset(
        dataroot="/home/mohammad/Documents/Gender_classification_train_version/new_dataset/train",
        image_transforms=tr)

    data_validation = AbolfazlDataset(
        dataroot="/home/mohammad/Documents/Gender_classification_train_version/new_dataset/val",
        image_transforms=tr)

    X = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=4)
    Y = torch.utils.data.DataLoader(data_validation, batch_size=batch_size, shuffle=True, num_workers=4)

    print("model Architector :\n")
    print(myNet)
    print('*********************\n')

    Train_acc, Val_acc = list(), list()
    Train_loss, Val_loss = list(), list()

    best_valid_loss = float('inf')
    for epoch in range(number_of_epochs):
        train_loss, train_acc = train(myNet, X, optimizer, criterion, device)
        valid_loss, valid_acc = evaluate(myNet, Y, criterion, device)

        if valid_loss < best_valid_loss:
            torch.save(myNet.state_dict(),
                       f"/home/mohammad/Documents/Gender_classification_train_version/weight/new_train3/Abolfazl{epoch + 1}.pt")
            model_scripted = torch.jit.script(myNet)  # for c++ infrence
            model_scripted.save(
                f"/home/mohammad/Documents/Gender_classification_train_version/weight/new_train3/{epoch + 1}_torch_script.pt")

        else:
            torch.save(myNet,
                       f"/home/mohammad/Documents/Gender_classification_train_version/weight/new_train3/{epoch + 1}_last.pth")
            model_scripted = torch.jit.script(myNet)  # for c++ infrence
            model_scripted.save(
                f"/home/mohammad/Documents/Gender_classification_train_version/weight/new_train3/{epoch + 1}_torch_script_last.pt")

        print(f'Epoch: {epoch + 1:02}')
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%")
        print(f"\tVal loss: {valid_loss:.3f} | val Acc: {valid_acc * 100:.2f}%")
        Train_acc.append(train_acc * 100)
        Val_acc.append(valid_acc * 100)
        Train_loss.append(train_loss)
        Val_loss.append(valid_loss)

    x = [i + 1 for i in range(number_of_epochs)]

    plt.plot(np.array(x), np.array(Train_acc), label='train')
    plt.plot(np.array(x), np.array(Val_acc), label='val')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
    plt.plot(np.array(x), np.array(Train_loss), label='train')
    plt.plot(np.array(x), np.array(Val_loss), label='val')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":
    main()
