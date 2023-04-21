import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torchvision.models import alexnet, resnet18, vgg16
from sklearn.metrics import balanced_accuracy_score
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve


LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: torch.cat([x, x, x], 0)),
    transforms.Resize(size=(224, 224)),
    transforms.Normalize(mean=[0.1706, 0.1706, 0.1706], std=[
                         0.2112, 0.2112, 0.2112])
])


class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)

        self.annot['Severity_Label'] = [LABELS_Severity[drss]
                                        for drss in copy.deepcopy(self.annot['DRSS'].values)]
        print(self.annot)
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        self.subset = subset
        self.nb_classes = len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)
        idx_each_class = [[] for i in range(self.nb_classes)]

    def __getitem__(self, index):

        img, target = Image.open(
            self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self._labels)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type=str,
                        default='df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type=str,
                        default='df_prime_test.csv')
    parser.add_argument('--data_root', type=str, default='')
    return parser.parse_args()


if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=100, shuffle=True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=True)

    lr = 1e-3
    epochs = 50

# initliaze the network
    num_classes = 3
    net = resnet18(pretrained=True)
    num_features = net.fc.in_features
    net.fc = nn.Linear(num_features, num_classes)

    net.to('cuda')

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    for epoch in range(epochs):
        net.train()  # training mode
        for iteration, (x, y) in enumerate(trainloader):
            optimizer.zero_grad()
            out = net(x.to('cuda'))
            loss = loss_function(out, y.to('cuda'))
            loss.backward()
            optimizer.step()
        print('Epoch : {} | Training Loss : {:0.4f}'.format(epoch, loss.item()))

    net.eval()  # testing mode
    total_correct = 0
    total_samples = 0
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for x, y in testloader:
            out = net(x.to('cuda'))
            preds = torch.argmax(out, axis=1)
            total_correct += torch.sum(preds == y.to('cuda'))
            total_samples += len(y)
            all_targets.extend(y.tolist())
            all_preds.extend(preds.tolist())
    test_accuracy = float(total_correct) / total_samples
    print('Test Accuracy : {:0.4f}'.format(test_accuracy))
    accuracy = accuracy_score(all_targets, all_preds)
    print('Accuracy : {:0.4f}'.format(accuracy))

    balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
    print('Balanced Accuracy : {:0.4f}'.format(balanced_accuracy))

    balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
    print('Balanced Accuracy : {:0.4f}'.format(balanced_accuracy))
    precision = precision_score(all_targets, all_preds, average='macro')
    recall = recall_score(all_targets, all_preds, average='macro')
    f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    print(f"Accuracy: {balanced_accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("Confusion matrix:")
    print(cm)
