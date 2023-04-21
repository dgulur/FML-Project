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
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import os
import copy
from pathlib import Path
from argparse import Namespace
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

LABELS_Severity = {35: 0,
                   43: 0,
                   47: 1,
                   53: 1,
                   61: 2,
                   65: 2,
                   71: 2,
                   85: 2}


mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize,
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


def input_output_generator(data_obj):
    data_obj_array = np.zeros((len(data_obj), 1, 224, 224))
    data_label_array = np.zeros(len(data_obj))
    for i in range(0, len(data_obj)):
        data_obj_array[i], data_label_array[i] = data_obj.__getitem__(i)
    return data_obj_array, data_label_array


batch_size = 100
if __name__ == '__main__':
    print("Current working directory:", os.getcwd())
    args = parse_args()
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)


    # create a DataLoader object for the test set
    testloader = DataLoader(testset, batch_size=batch_size)

    # create a DataLoader object for the training set
    trainloader = DataLoader(trainset, batch_size=batch_size)


    # define an empty list to store the data and label arrays
    data_obj_array_test = []
    data_label_array_test = []

    # iterate over the batches in the test loader
    for batch in testloader:
        # append the data and label arrays to the list
        data_obj_batch, data_label_batch = batch
        data_obj_array_test.append(data_obj_batch.numpy())
        data_label_array_test.append(data_label_batch.numpy())

    # concatenate the data and label arrays
    data_obj_array_test = np.concatenate(data_obj_array_test, axis=0)
    data_label_array_test = np.concatenate(data_label_array_test, axis=0)

    # define an empty list to store the data and label arrays
    data_obj_array_train = []
    data_label_array_train = []

    # iterate over the batches in the training loader
    for batch in trainloader:
        # append the data and label arrays to the list
        data_obj_batch, data_label_batch = batch
        data_obj_array_train.append(data_obj_batch.numpy())
        data_label_array_train.append(data_label_batch.numpy())

    # concatenate the data and label arrays
    data_obj_array_train = np.concatenate(data_obj_array_train, axis=0)
    data_label_array_train = np.concatenate(data_label_array_train, axis=0)



    train_images = data_obj_array_train.reshape(
        data_obj_array_train.shape[0], -1)
    test_images = data_obj_array_test.reshape(data_obj_array_test.shape[0], -1)


    clf = LogisticRegression(max_iter=10000, penalty='l2', C=1.0)
    clf.fit(train_images, data_label_array_train)
    y_pred = clf.predict(test_images)
   

    balanced_accuracy = balanced_accuracy_score(data_label_array_test, y_pred)
    # Calculate the evaluation metrics of the classifier
    acc = accuracy_score(data_label_array_test, y_pred)

    # Print the evaluation metrics of the classifier
    print(f"Accuracy: {acc:.3f}")
    print(f"Accuracy: {balanced_accuracy:.3f}")
    precision = precision_score(data_label_array_test, y_pred, average='macro')
    recall = recall_score(data_label_array_test, y_pred, average='macro')
    f1 = f1_score(data_label_array_test, y_pred, average='macro')
    cm = confusion_matrix(data_label_array_test, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("Confusion matrix:")
    print(cm)

