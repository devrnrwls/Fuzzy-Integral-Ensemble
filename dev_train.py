

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
# from probability_extraction import *
import argparse

import torch

import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

# mean = np.array([0.20369699, 0.22522233, 0.22994686])
# std = np.array([0.14865874, 0.17049677, 0.14530116])

mean = np.array([0.734, 0.674, 0.643])
std = np.array([0.187, 0.231, 0.239])


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


def plot(val_loss, train_loss, typ):
    plt.title("{} after epoch: {}".format(typ, len(train_loss)))
    plt.xlabel("Epoch")
    plt.ylabel(typ)
    plt.plot(list(range(len(train_loss))), train_loss, color="r", label="Train " + typ)
    plt.plot(list(range(len(val_loss))), val_loss, color="b", label="Validation " + typ)
    plt.legend()
    plt.savefig(os.path.join(data_dir, typ + ".png"))
    #     plt.figure()
    plt.close()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25, model_name="kaggle"):
    val_loss_gph = []
    train_loss_gph = []
    val_acc_gph = []
    train_acc_gph = []
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # 네트워크 별로 따른 데이터셋 학습
            if category == "":
                category_phase = phase
            else:
                category_phase = category + "_"+ phase

            for inputs, labels in dataloaders[category_phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs,
                                         1)  # was (outputs,1) for non-inception and (outputs.data,1) for inception
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_gph.append(epoch_loss)
                train_acc_gph.append(epoch_acc.cpu())
            if phase == 'val':
                val_loss_gph.append(epoch_loss)
                val_acc_gph.append(epoch_acc.cpu())

            plot(val_loss_gph, train_loss_gph, model_name + "_Loss")
            plot(val_acc_gph, train_acc_gph, model_name + "Accuracy")

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc >= best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), data_dir + "/" + model_name + ".pth")
                print('==>Model Saved')

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def get_Laplacian(images):
    # mean = np.array([0.485, 0.456, 0.406])
    # std = np.array([0.229, 0.224, 0.225])

    image_np = images.squeeze().numpy()

    # 이미지 데이터를 0-255 범위로 스케일링
    image_np = (image_np * std[:, None, None]) + mean[:, None, None]
    image_np = (image_np * 255).astype(np.uint8)
    image_np = np.transpose(image_np, (1, 2, 0))

    grey = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

    var = cv2.Laplacian(grey, cv2.CV_64F).var()

    return var


# Getting Proba distribution
def get_probability(image_datasets, model, data_dir, model_name, file_name):
    print("\nGetting the Probability Distribution")

    #정답이 안섞이고 COVID, non-COVID 순으로 나올 수 있게 shuttle을 사용하지 않음
    testloader = torch.utils.data.DataLoader(image_datasets[file_name], batch_size=1)

    model = model.eval()

    correct = 0
    total = 0
    import csv
    import numpy as np
    # f = open(data_dir + '/' + model_name + ".csv", 'w+', newline='')
    f = open(data_dir + '/' + model_name +"_" +file_name+"_Model.csv", 'w+', newline='')
    writer = csv.writer(f)

    #blurr data save file
    f_blurr = open(data_dir + '/' + model_name +"_" +file_name+"_Lap.csv", 'w+', newline='')
    writer_blurr = csv.writer(f_blurr)

    with torch.no_grad():
        num = 0
        temp_array = np.zeros((len(testloader), num_classes))
        temp_array_blurr = np.zeros((len(testloader), 1))

        for data in testloader:
            images, labels = data
            labels = labels.cuda()
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
            prob = torch.nn.functional.softmax(outputs, dim=1)
            temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
            temp_array_blurr[num] = np.asarray(get_Laplacian(images))
            num += 1
    print("Accuracy = ", 100 * correct / total)

    for i in range(len(testloader)):
        writer.writerow(temp_array[i].tolist())
        writer_blurr.writerow(temp_array_blurr[i].tolist())
    f.close()
    f_blurr.close()

parser = argparse.ArgumentParser()
parser.add_argument('--data_directory', type=str, required = True, help='Directory where data is stored')
parser.add_argument('--epochs', type=int, default = 25, help='Number of epochs to run the models')
args = parser.parse_args()

data_dir = args.data_directory

# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'blurred_val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    'brightened_val': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    'darkened_val': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    'blurred_train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    'brightened_train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    'darkened_train': transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'blurred_val', 'brightened_val', 'darkened_val', 'blurred_train', 'brightened_train', 'darkened_train']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val', 'blurred_val', 'brightened_val', 'darkened_val', 'blurred_train', 'brightened_train', 'darkened_train']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'blurred_val', 'brightened_val', 'darkened_val', 'blurred_train', 'brightened_train', 'darkened_train']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(class_names)

#Get probability distributions from the 4 models
num_epochs = args.epochs

criterion = nn.CrossEntropyLoss()

#Is_ensemble 조절해서 여러 모델을 서로 다른 학습 데이터로 학습할지 결정
#true = 서로 다른 데이터 셋으로 4개 네트워크 학습
#false = 동일한 데이터 셋으로 4개 네트워크 학습
Is_ensemble = False
if Is_ensemble:
    #dictionaly value 입력하시요!! >> "" "blurred" "brightened" "darkened"
    category_model = {"Kaggle_vgg11":'', "Kaggle_googlenet": '',
                         "Kaggle_squeezenet": '', "Kaggle_wideresnet": ''}
else:
    #1개의 데이터셋(category)으로 모든 네트워크 학습 진행
    # 오리지널 train, val = "" 입력
    category = "" # select>> "" "blurred" "brightened" "darkened"
    category_model = {"Kaggle_vgg11": category, "Kaggle_googlenet": category,
                      "Kaggle_squeezenet": category, "Kaggle_wideresnet": category}

# category = "darkened"
model_name = "Kaggle_vgg11"
category = category_model["Kaggle_vgg11"] #return dictionaly value
model = models.vgg11_bn(pretrained = True)
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_vgg11')

category = category_model["Kaggle_googlenet"]
model_name = "Kaggle_googlenet"
model = models.googlenet(pretrained = True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_googlenet')

category = category_model["Kaggle_squeezenet"]
model_name = "Kaggle_squeezenet"
model = models.squeezenet1_1(pretrained = True)
model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_squeezenet')

category = category_model["Kaggle_wideresnet"]
model_name = "Kaggle_wideresnet"
model = models.wide_resnet50_2(pretrained = True)
# model = models.wide_resnet50_2
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)

model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_wideresnet')

# get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11', file_name='val')
# get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11', file_name='blurred_val')
# get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11', file_name='brightened_val')
# get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11', file_name='darkened_val')