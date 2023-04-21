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




num_classes = 3
model = resnet18(pretrained=True).cuda()
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
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
        self.patient_id = self.annot['Patient_ID']
        self.eye_id = self.annot['Eye_ID']
        self.week_num = self.annot['Week_Num']
        self.eye_side = np.where(self.annot['Eye_side'] == 'OD', 1, np.where(self.annot['Eye_side'] == 'OS', 2, self.annot['Eye_side']))
        self.eye_data = np.where(self.annot['Eye_Data'] == 'OD', 1, np.where(self.annot['Eye_Data'] == 'OS', 2, self.annot['Eye_Data']))
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
        
        self.metadata = np.array([self.patient_id,self.eye_id,self.week_num,self.eye_side,self.eye_data,self.frame_num,self.age,self.gender,self.race,self.diabetes_type,self.diabetes_years,self.bmi,self.bcva,self.cst,self.leakage_index]).T + 0.001
        assert len(self.path_list) == len(self._labels)  

    def __getitem__(self, index):
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]
        
        
        if self.transform is not None:
            img = self.transform(img)
        img = img.numpy()
        metadata = self.metadata[index].reshape((1, 15, 1))
        metadata = np.pad(metadata, ((0,0),(104,105),(0,0)), mode='constant')      
        new_arr = np.zeros((3, 224, 224))
        new_arr[:, 112, :] = metadata.squeeze()
        metadata = np.tile(new_arr[:, 112, :], (224, 1, 1))
        metadata = np.transpose(metadata, (1, 0, 2))  
        img = img + metadata
        img = torch.tensor(img).float()
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

    accuracy = np.mean(y_true == y_pred)
    balanced_accuracy = bal_acc = balanced_accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"Balanced Accuracy: {balanced_accuracy}")

