import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models,transforms
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
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve

warnings.filterwarnings("ignore")

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
    transforms.Normalize(mean=[0.1706, 0.1706, 0.1706], std=[0.2112, 0.2112, 0.2112])
])

# Load the pre-trained AlexNet model and move it to GPU
model = models.alexnet(pretrained=True).cuda()
# Remove the last layer of the model
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
# Set the model to evaluation mode
model.eval()

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
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)  

    def __getitem__(self, index):
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target
        
    def __len__(self):
        return len(self._labels)         

if __name__ == '__main__':
    trainset = OCTDataset(args, 'train', transform=transform)
    testset = OCTDataset(args, 'test', transform=transform)
    
    # Define the data loader for the dataset
    data_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    
    # Extract the features for each image in the dataset and save them
    features = []
    labels = []
    for images, targets in data_loader:
    
        # Move the images and targets to GPU
        images = images.cuda()
        targets = targets.cuda()
        # Apply the image transforms and pass the images through the model
        with torch.no_grad():
            outputs = model(images)
        # Extract the features from the output of the model
        features_batch = outputs.detach().cpu().numpy()
        # Append the features and labels to the lists
        features.append(features_batch)
        labels.append(targets.cpu().numpy())
    # Concatenate the features and labels into a single numpy array
    features = np.concatenate(features)
    labels = np.concatenate(labels)

    clf = LogisticRegression(max_iter=10000)
    clf.fit(features,labels)
    
    # Evaluate the classifier on a test set
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    y_pred = []
    y_true = []
    for images, targets in test_loader:
        images = images.cuda()
        targets = targets.cuda()
        # Apply the image transforms and pass the images through the model
        with torch.no_grad():
            outputs = model(images)
        # Extract the features from the output of the model
        features_batch = outputs.detach().cpu().numpy()
        # Predict the labels using the trained classifier
        y_pred_batch = clf.predict(features_batch)
        # Append the predicted and true labels to the lists
        y_pred.append(y_pred_batch)
        y_true.append(targets.cpu().numpy())
    # Concatenate the predicted and true labels into a single numpy array
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)

    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1-score: {f1:.3f}")
    print("Confusion matrix:")
    print(cm)
