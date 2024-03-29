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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve
from scipy.stats import entropy

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
            self.annot_x = pd.read_csv(args.annot_train_prime)
            self._35 = self.annot_x[self.annot_x['DRSS']==35]
            self._43 = self.annot_x[self.annot_x['DRSS']==43]
            self._zero = pd.concat([self._35, self._43])
            metadata = np.array([self._zero['Patient_ID'],self._zero['Eye_ID'],self._zero['Week_Num'],
                                 self._zero['Frame_Num'],self._zero['Age'],self._zero['Gender'],
                                 self._zero['Race'],self._zero['Diabetes_Type'],self._zero['Diabetes_Years'],
                                 self._zero['BMI'],self._zero['BCVA'],self._zero['CST'],self._zero['Leakage_Index']])
            metadata = metadata.astype(float).T + 1
            row_entropies = np.apply_along_axis(entropy, axis=1, arr=metadata)
            print(row_entropies)
            # Sort the samples in decreasing order of entropy and select the top ones
            selected_indices = np.argsort(row_entropies)[::-1][:4000]
            print(selected_indices)
            self._zero = self._zero.iloc[selected_indices]
            

            self._47 = self.annot_x[self.annot_x['DRSS']==47]
            self._53 = self.annot_x[self.annot_x['DRSS']==53]
            self._one = pd.concat([self._47, self._53])
            
            metadata = np.array([self._one['Patient_ID'],self._one['Eye_ID'],self._one['Week_Num'],
                                 self._one['Frame_Num'],self._one['Age'],self._one['Gender'],
                                 self._one['Race'],self._one['Diabetes_Type'],self._one['Diabetes_Years'],
                                 self._one['BMI'],self._one['BCVA'],self._one['CST'],self._one['Leakage_Index']])
            metadata = metadata.astype(float).T + 1
            row_entropies = np.apply_along_axis(entropy, axis=1, arr=metadata)
            print(row_entropies)
            # Sort the samples in decreasing order of entropy and select the top ones
            selected_indices = np.argsort(row_entropies)[::-1][:4000]
            print(selected_indices)
            
            self._one = self._one.iloc[selected_indices]
            
            self._61 = self.annot_x[self.annot_x['DRSS']==61]
            self._65 = self.annot_x[self.annot_x['DRSS']==65]
            self._two_1 = pd.concat([self._61, self._65])
            
            self._71 = self.annot_x[self.annot_x['DRSS']==71]
            self._85 = self.annot_x[self.annot_x['DRSS']==85]
            self._two_2 = pd.concat([self._71, self._85])
            
            self._two = pd.concat([self._two_1, self._two_2])
            metadata = np.array([self._two['Patient_ID'],self._two['Eye_ID'],self._two['Week_Num'],
                                 self._two['Frame_Num'],self._two['Age'],self._two['Gender'],
                                 self._two['Race'],self._two['Diabetes_Type'],self._two['Diabetes_Years'],
                                 self._two['BMI'],self._two['BCVA'],self._two['CST'],self._two['Leakage_Index']])
            metadata = metadata.astype(float).T + 1
            row_entropies = np.apply_along_axis(entropy, axis=1, arr=metadata)
            print(row_entropies)
            # Sort the samples in decreasing order of entropy and select the top ones
            selected_indices = np.argsort(row_entropies)[::-1][:4000]
            print(selected_indices)
            
            self._two = self._two.iloc[selected_indices]
            self._annot_0 = pd.concat([self._zero, self._one])
            self.annot = pd.concat([self._annot_0, self._two])
            
        elif subset == 'test':
            self.annot = pd.read_csv(args.annot_test_prime)
        
        self.annot['Severity_Label'] = [LABELS_Severity[drss] for drss in copy.deepcopy(self.annot['DRSS'].values)] 
        self.root = os.path.expanduser(args.data_root)
        self.transform = transform
        self.nb_classes=len(np.unique(list(LABELS_Severity.values())))
        self.path_list = self.annot['File_Path'].values
        self._labels = self.annot['Severity_Label'].values
        assert len(self.path_list) == len(self._labels)  
        
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
        
        metadata = np.array([self.patient_id,self.eye_id,self.week_num,self.eye_side,self.eye_data,self.frame_num,self.age,self.gender,self.race,self.diabetes_type,self.diabetes_years,self.bmi,self.bcva,self.cst,self.leakage_index])
        metadata = metadata.astype(float).T
        
        # compute the mean and standard deviation along each dimension
        mean = np.mean(metadata, axis=0)
        std = np.std(metadata, axis=0) + 0.001
        # normalize the array by subtracting the mean and dividing by the standard deviation

        metadata = (metadata-mean)/std
        
        if subset == 'train':
            self.metadata = metadata
        else:
            self.metadata = np.zeros(metadata.shape)

    def __getitem__(self, index):
        img, target = Image.open(self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)
        
        metadata_info = self.metadata[index]
        
        return img, target,metadata_info

    def __len__(self):
        return len(self._labels)

class SiameseNetwork(nn.Module):
    def __init__(self, num_metadata_features, num_classes):
        super(SiameseNetwork, self).__init__()
        # Image branch
        self.image_branch = models.resnet18(pretrained=True)
        num_features = self.image_branch.fc.in_features
        self.image_branch.fc = nn.Linear(num_features, num_classes)
        
        # Metadata branch
        self.metadata_branch = nn.Sequential(
            nn.Linear(num_metadata_features, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    # Output layer
        self.output_layer = nn.Linear(2*num_classes, num_classes).to("cuda")


    def forward(self, image, metadata):
        # Pass image through the image branch
        image_out = self.image_branch(image)
        # Pass metadata through the metadata branch
        metadata_out = self.metadata_branch(metadata)
        combined = torch.cat((image_out, metadata_out), dim=1).to("cuda")
        output = self.output_layer(combined)
        return output



# Example usage:
num_metadata_features = 15
num_classes = 3
if torch.cuda.is_available():
    model = SiameseNetwork(num_metadata_features, num_classes).to("cuda")
else:
    model = SiameseNetwork(num_metadata_features, num_classes).to("cpu")
    
# Train the model with both images and metadata
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if torch.cuda.is_available():
    class_weights = torch.tensor([1.0380071905495634, 0.6874149659863945, 1.7185374149659864], dtype=torch.float32).to('cuda')
else:
    class_weights = torch.tensor([1.0380071905495634, 0.6874149659863945, 1.7185374149659864], dtype=torch.float32).to('cpu')
loss_function = nn.CrossEntropyLoss()

trainset = OCTDataset(args, 'train', transform=transform)
testset = OCTDataset(args, 'test', transform=transform)
print(len(trainset))
print(len(testset))

# Define the data loader for the dataset
data_loader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # Get the inputs and labels from the data loader
        inputs, labels, metadata = data
        #print(inputs)
        #print(labels)
        #print(metadata)
        mean_m = np.nanmean(metadata)
        # replace NaN values with the mean
        metadata[np.isnan(metadata)] = 0.001
        metadata = torch.tensor(metadata,dtype=torch.float32)
        # Move the inputs and labels to GPU
        if torch.cuda.is_available():
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
            metadata = metadata.to("cuda")
        else:
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")
            metadata = metadata.to("cpu")
            
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward + backward + optimize
        outputs = model.forward(inputs,metadata)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        # Print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # Print0 every 100 mini-batches
            print('[Epoch %d, Batch %5d] loss: %.3f' %
                (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

model.eval()  # testing mode
total_correct = 0
total_samples = 0
all_targets = []
all_preds = []
with torch.no_grad():
    for x, y, z in testloader:
        z[np.isnan(z)] = 1e-10
        z = torch.tensor(z,dtype=torch.float32)
        if torch.cuda.is_available():
            out = model(x.to('cuda'),z.to('cuda'))
        else:
            out = model(x.to('cpu'),z.to('cpu'))
        preds = torch.argmax(out, axis=1)
        if torch.cuda.is_available():
            total_correct += torch.sum(preds == y.to('cuda'))
        else:
            total_correct += torch.sum(preds == y.to('cpu'))
        total_samples += len(y)
        all_targets.extend(y.tolist())
        all_preds.extend(preds.tolist())
test_accuracy = float(total_correct) / total_samples
print('Test Accuracy : {:0.4f}'.format(test_accuracy))
balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
print('Balanced Accuracy : {:0.4f}'.format(balanced_accuracy))

balanced_accuracy = balanced_accuracy_score(all_targets, all_preds)
# Calculate the evaluation metrics of the classifier
acc = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, average='macro')
recall = recall_score(all_targets, all_preds, average='macro')
f1 = f1_score(all_targets, all_preds, average='macro')
cm = confusion_matrix(all_targets, all_preds)
print(f"Accuracy: {acc:.3f}")
print(f"Accuracy: {balanced_accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print("Confusion matrix:")
print(cm)
