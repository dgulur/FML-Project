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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annot_train_prime', type=str,
                        default='df_prime_train.csv')
    parser.add_argument('--annot_test_prime', type=str,
                        default='df_prime_test.csv')
    parser.add_argument('--data_root', type=str, default='')
    return parser.parse_args()

args = Namespace(**args_dict)


class OCTDataset(Dataset):
    def __init__(self, args, subset='train', transform=None,):
        if subset == 'train':
            self.annot = pd.read_csv(args.annot_train_prime)
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
        self.annot['Severity_Label'] = [LABELS_Severity[drss]
                                        for drss in copy.deepcopy(self.annot['DRSS'].values)]
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        self.nb_classes = len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values

        self.patient_id = self.annot['Patient_ID']
        self.eye_id = self.annot['Eye_ID']
        self.week_num = self.annot['Week_Num']
        self.eye_side = np.where(self.annot['Eye_side'] == 'OD', 1, np.where(
            self.annot['Eye_side'] == 'OS', 2, self.annot['Eye_side']))
        self.eye_data = np.where(self.annot['Eye_Data'] == 'OD', 1, np.where(
            self.annot['Eye_Data'] == 'OS', 2, self.annot['Eye_Data']))
        self.frame_num = self.annot['Frame_Num']
        self.age = self.annot['Age']
        self.gender = self.annot['Gender']
        self.race = self.annot['Race']
        self.diabetes_type = self.annot['Diabetes_Type']
        self.diabetes_years = self.annot['Diabetes_Years']
        self.bmi = self.annot['BMI']
        self.bcva = self.annot['BCVA']
        self.cst = self.annot['CST']
        self.leakage_index = self.annot['Leakage_Index']

        self.metadata = np.array([self.patient_id, self.eye_id, self.week_num, self.eye_side, self.eye_data, self.frame_num, self.age,
                                 self.gender, self.race, self.diabetes_type, self.diabetes_years, self.bmi, self.bcva, self.cst, self.leakage_index]).T + 0.001
        assert len(self.path_list) == len(self._labels)
        self.features = self.extract_features()

    def __getitem__(self, index):
        img, target = Image.open(
            self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def extract_features(self):
        features = []
        labels = []
        for i in range(len(self.path_list)):
            img = Image.open(self.root+self.path_list[i]).convert("L")
            if self.transform is not None:
                img = self.transform(img)
            img = img.numpy().ravel()
            img = list(img) + list(self.metadata[i])
            features.append(img)
        features = np.array(features)
        row_means = np.nanmean(features, axis=1)
        inds = np.where(np.isnan(features))
        features[inds] = np.take(row_means, inds[0])
        return features

    def __len__(self):
        return len(self._labels)


if __name__ == '__main__':
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    print(trainset[1][0].shape)
    print(trainset[1][1].shape)
    print(len(trainset), len(testset))
    # Train the Naive Bayes classifier

    clf = LogisticRegression(max_iter=10000, penalty='l2', C=1.0)
    clf.fit(trainset.features, trainset._labels)

    y_pred = clf.predict(testset.features)
    

    balanced_accuracy = balanced_accuracy_score(testset._labels, y_pred)
    # Calculate the evaluation metrics of the classifier
    acc = accuracy_score(testset._labels, y_pred)

    # Print the evaluation metrics of the classifier
    print(f"Accuracy: {acc:.3f}")
    print(f"Accuracy: {balanced_accuracy:.3f}")
    precision = precision_score(testset._labels, y_pred, average='macro')
    recall = recall_score(testset._labels, y_pred, average='macro')
    f1 = f1_score(testset._labels, y_pred, average='macro')
    cm = confusion_matrix(testset._labels, y_pred)

    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("Confusion matrix:")
    print(cm)
