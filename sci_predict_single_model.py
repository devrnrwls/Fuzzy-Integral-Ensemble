import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.metrics import *
import pandas as pd
import numpy as np
from sklearn.metrics import *
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2


def metrics(labels, predictions, classes):
    print("Classification Report:")
    print(classification_report(labels, predictions, target_names=classes, digits=4))
    matrix = confusion_matrix(labels, predictions)
    print("Confusion matrix:")
    print(matrix)
    print("\nClasswise Accuracy :{}".format(matrix.diagonal() / matrix.sum(axis=1)))
    print("\nBalanced Accuracy Score: ", balanced_accuracy_score(labels, predictions))

    # Confusion matrix visualization
    plt.figure(figsize=(len(classes), len(classes)))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def predicting(ensemble_prob):
    prediction = np.zeros((ensemble_prob.shape[0],))
    for i in range(ensemble_prob.shape[0]):
        temp = ensemble_prob[i]
        t = np.where(temp == np.max(temp))[0][0]
        prediction[i] = t
    return prediction


def get_metrics(work_category, image_datasets, class_names, num_classes, model, model_name, incorrect_image_check):
    testloader = torch.utils.data.DataLoader(image_datasets[work_category], batch_size=1)
    # model = model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        num = 0
        temp_array = np.zeros((len(testloader), num_classes))
        temp_labels = np.zeros((len(testloader),), dtype=int)
        for data in testloader:
            images, labels = data
            labels = labels.cuda()
            outputs = model(images.cuda())
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels.cuda()).sum().item()
            prob = torch.nn.functional.softmax(outputs, dim=1)
            temp_array[num] = np.asarray(prob[0].tolist()[0:num_classes])
            temp_labels[num] = labels.item()
            num += 1

            if incorrect_image_check:
                if predicted != labels.item():
                    print(class_names)
                    print(outputs)
                    img_array = images[0].cpu().numpy().transpose(1, 2, 0)
                    img_array = img_array * std + mean
                    # 이미지를 Matplotlib을 사용하여 출력
                    plt.imshow(img_array)
                    plt.title(f"Predicted: {class_names[predicted.item()]}, Actual: {class_names[labels.item()]}")
                    plt.show()

    print("Accuracy = ", 100 * correct / total)

    pred_result = predicting(temp_array)
    metrics(temp_labels, pred_result, class_names)

if __name__ == '__main__':

    data_dir = "./packaging/"
    work_category = 'val'
    incorrect_image_check = True

    #Kaggle_vgg11 Kaggle_googlenet Kaggle_squeezenet Kaggle_wideresnet
    model_dir = "Result/train_result/"
    model_name = 'Kaggle_googlenet'

    mean = np.array([0.957, 0.900, 0.852])
    std = np.array([0.126, 0.215, 0.266])

    data_transforms = {
        work_category : transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in [work_category]}

    class_names = image_datasets[work_category].classes
    num_classes = len(class_names)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_dir_modelName = model_dir + model_name

    if model_name == "Kaggle_vgg11":
        model = models.vgg11_bn(pretrained=False)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir_modelName, model_name + '.pth')))
        model.eval()

    elif model_name == "Kaggle_googlenet":
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir_modelName, model_name + '.pth')))
        print(model)
        model.eval()

    elif model_name == "Kaggle_squeezenet":
        model = models.squeezenet1_1(pretrained=False)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir_modelName, model_name + '.pth')))
        model.eval()

    elif model_name == "Kaggle_wideresnet":
        model = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir_modelName, model_name + '.pth')))
        model.eval()

    get_metrics(work_category, image_datasets, class_names, num_classes, model, model_name, incorrect_image_check)