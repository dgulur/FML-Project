import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import models, transforms
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
from sklearn.decomposition import PCA
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

mean = (.1706)
std = (.2112)
normalize = transforms.Normalize(mean=mean, std=std)

transform = transforms.Compose([
    transforms.Resize(size=(224, 224)),
    transforms.ToTensor(),
    normalize,
])

args_dict = {'annot_train_prime': r"/storage/home/hpaceice1/dgulur3/df_prime_train.csv",
             'annot_test_prime': r"/storage/home/hpaceice1/dgulur3/df_prime_test.csv",
             'data_root': r'/storage/home/hpaceice1/shared-classes/materials/ece8803fml/'}

args = Namespace(**args_dict)


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


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
        assert len(self.path_list) == len(self._labels)

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

        metadata = np.array([self.patient_id, self.eye_id, self.week_num, self.eye_side, self.eye_data, self.frame_num, self.age,
                            self.gender, self.race, self.diabetes_type, self.diabetes_years, self.bmi, self.bcva, self.cst, self.leakage_index])
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
        img, target = Image.open(
            self.root+self.path_list[index]).convert("L"), self._labels[index]

        if self.transform is not None:
            img = self.transform(img)

            img1 = img.numpy().squeeze()
           

            pca = PCA(n_components=50)

            img_reduced = pca.fit_transform(img1)

            img_reduced = img_reduced.reshape(224, 50)

            torch_tensor = torch.from_numpy(img_reduced)

            torch_tensor = torch.unsqueeze(torch_tensor, dim=0)
            
            img_final = torch.cat((torch_tensor,torch_tensor, torch_tensor), dim=0)
           

        metadata_info = self.metadata[index]

        return img_final, target, metadata_info

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

    def forward(self, image, metadata):
        # Pass image through the image branch
        image_out = self.image_branch(image)
        # Pass metadata through the metadata branch
        metadata_out = self.metadata_branch(metadata)
        # Concatenate the outputs from the two branches
        combined = torch.cat((image_out, metadata_out), dim=1)
        return combined


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
    class_weights = torch.tensor([1.0380071905495634, 0.6874149659863945, 1.7185374149659864,
                                 1.0380071905495634, 0.6874149659863945, 1.718537414965986], dtype=torch.float32).to('cuda')
else:
    class_weights = torch.tensor([1.0380071905495634, 0.6874149659863945, 1.7185374149659864,
                                 1.0380071905495634, 0.6874149659863945, 1.718537414965986], dtype=torch.float32).to('cpu')


loss_function = nn.CrossEntropyLoss()

trainset = OCTDataset(args, 'train', transform=transform)
testset = OCTDataset(args, 'test', transform=transform)
print(type(trainset[1][0]))
print(type(trainset[1][1]))
print(type(trainset[1][2]))

# Define the data loader for the dataset
data_loader = torch.utils.data.DataLoader(
    trainset, batch_size=100, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True)

num_epochs = 25
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # Get the inputs and labels from the data loader
        inputs, labels, metadata = data
        # print(inputs)
        # print(labels)
        # print(metadata)
        mean_m = np.nanmean(metadata)
        # replace NaN values with the mean
        metadata[np.isnan(metadata)] = mean_m
        metadata = torch.tensor(metadata, dtype=torch.float32)
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
        outputs = model.forward(inputs, metadata)
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
        z = torch.tensor(z, dtype=torch.float32)
        if torch.cuda.is_available():
            out = model(x.to('cuda'), z.to('cuda'))
        else:
            out = model(x.to('cpu'), z.to('cpu'))
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
