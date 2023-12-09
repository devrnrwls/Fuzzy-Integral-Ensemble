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

data_dir = "./data"
num_epochs = 5

mean = np.array([0.734, 0.674, 0.643])
std = np.array([0.187, 0.231, 0.239])

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

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
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

            plot(val_loss_gph, train_loss_gph, "Loss")
            plot(val_acc_gph, train_acc_gph, "Accuracy")

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

def get_probability(image_datasets, model, data_dir, model_name):
    print("\nGetting the Probability Distribution")
    testloader = torch.utils.data.DataLoader(image_datasets['val'], batch_size=1)

    model = model.eval()

    correct = 0
    total = 0
    import csv
    import numpy as np
    f = open(data_dir + '/' + model_name + ".csv", 'w+', newline='')
    writer = csv.writer(f)

    with torch.no_grad():
        num = 0
        temp_array = np.zeros((len(testloader), num_classes))
        for data in testloader:
            images, labels = data
            labels = labels.cuda()
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
            prob = torch.nn.functional.softmax(outputs, dim=1)
            temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
            num += 1
    print("Accuracy = ", 100 * correct / total)

    for i in range(len(testloader)):
        writer.writerow(temp_array[i].tolist())
    f.close()


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
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=16,
                                             shuffle=True, num_workers=0)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
num_classes = len(class_names)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(class_names)

criterion = nn.CrossEntropyLoss()
model = models.vgg11_bn(pretrained = True)
num_ftrs = model.classifier[0].in_features
model.classifier = nn.Linear(num_ftrs, num_classes)
model = model.to(device)
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum = 0.99)
step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma=0.1)
model = train_model(model, criterion, optimizer, step_lr_scheduler, num_epochs=num_epochs, model_name = 'Kaggle_vgg11')
get_probability(image_datasets,model,data_dir,model_name='Kaggle_vgg11')