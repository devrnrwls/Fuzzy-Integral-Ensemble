import argparse

#Perform the ensemble
from dev_ensemble import *
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import os
import cv2

mean = np.array([0.20369699, 0.22522233, 0.22994686])
std = np.array([0.14865874, 0.17049677, 0.14530116])

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
# parser.add_argument('--epochs', type=int, default = 25, help='Number of epochs to run the models')
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
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'blurred_val', 'brightened_val', 'darkened_val']}

class_names = image_datasets['train'].classes
num_classes = len(class_names)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Kaggle_vgg11 Kaggle_googlenet Kaggle_squeezenet Kaggle_wideresnet
model_name = "Kaggle_squeezenet"

if model_name == "Kaggle_vgg11":
    model = models.vgg11_bn(pretrained = True)
    num_ftrs = model.classifier[0].in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)
elif model_name == "Kaggle_googlenet":
    model = models.googlenet(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
elif model_name == "Kaggle_squeezenet":
    model = models.squeezenet1_1(pretrained = True)
    model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
elif model_name == "Kaggle_wideresnet":
    model = models.wide_resnet50_2(pretrained = True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

model = model.to(device)
model.load_state_dict(torch.load(os.path.join(data_dir, model_name +'.pth')))

get_probability(image_datasets,model,data_dir,model_name=model_name, file_name='val')
# get_probability(image_datasets,model,data_dir,model_name=model_name, file_name='blurred_val')
# get_probability(image_datasets,model,data_dir,model_name=model_name, file_name='brightened_val')
# get_probability(image_datasets,model,data_dir,model_name=model_name, file_name='darkened_val')

# prob1,labels = getfile(model_name+"_val_Model",root = data_dir)
# prob2,_ = getfile(model_name+"_brightened_val_Model",root = data_dir)
# prob3,_ = getfile(model_name+"_darkened_val_Model",root = data_dir)
# prob4,_ = getfile(model_name+"_blurred_val_Model",root = data_dir)

prob1_Lap = get_Lap_file(model_name+"_val_Lap",root = data_dir)
# prob2_Lap = get_Lap_file(model_name + "_brightened_val_Lap",root = data_dir)
# prob3_Lap = get_Lap_file(model_name + "_darkened_val_Lap",root = data_dir)
# prob4_Lap = get_Lap_file(model_name + "_blurred_val_Lap",root = data_dir)

prob1,labels = getfile(model_name+"_val_Model",root = data_dir)
prob2 = prob1
prob3 = prob1
prob4 = prob1

prob1_Lap = get_Lap_file(model_name+"_val_Lap",root = data_dir)
prob2_Lap = prob1_Lap
prob3_Lap = prob1_Lap
prob4_Lap = prob1_Lap

ensemble_sugeno(labels,prob1,prob2,prob3,prob4, prob1_Lap, prob2_Lap, prob3_Lap, prob4_Lap)

