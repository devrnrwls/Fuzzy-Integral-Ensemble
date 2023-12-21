#sample code: https://csm-kr.tistory.com/74

import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F
from datetime import datetime

class GradCam(nn.Module):
    def __init__(self, model, module, layer):
        super().__init__()
        self.model = model
        self.module = module
        self.layer = layer
        self.register_hooks()

    def register_hooks(self):
        for modue_name, module in self.model._modules.items():
            if modue_name == self.module:
                for layer_name, module in module._modules.items():
                    if layer_name == self.layer:
                        module.register_forward_hook(self.forward_hook)
                        module.register_backward_hook(self.backward_hook)

    def forward(self, input, target_index):
        outs = self.model(input)
        outs = outs.squeeze()  # [1, num_classes]  --> [num_classes]

        # 가장 큰 값을 가지는 것을 target index 로 사용
        if target_index is None:
            target_index = outs.argmax()

        outs[target_index].backward(retain_graph=True)
        a_k = torch.mean(self.backward_result, dim=(1, 2), keepdim=True)  # [512, 1, 1]
        out = torch.sum(a_k * self.forward_result, dim=0).cuda() # [512, 7, 7] * [512, 1, 1]
        out = torch.relu(out) / torch.max(out)  # 음수를 없애고, 0 ~ 1 로 scaling # [7, 7]
        out = F.upsample_bilinear(out.unsqueeze(0).unsqueeze(0), [224, 224])  # 4D로 바꿈
        return out.cpu().detach().squeeze().numpy()

    def forward_hook(self, _, input, output):
        self.forward_result = torch.squeeze(output)

    def backward_hook(self, _, grad_input, grad_output):
        self.backward_result = torch.squeeze(grad_output[0])


if __name__ == '__main__':

    def preprocess_image(img):
        means = np.array([0.957, 0.900, 0.852])
        stds = np.array([0.126, 0.215, 0.266])

        preprocessed_img = img.copy()[:, :, ::-1]
        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
            preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        preprocessed_img = torch.from_numpy(preprocessed_img)
        preprocessed_img.unsqueeze_(0)
        input = preprocessed_img.requires_grad_(True)
        return input

    def show_cam_on_image(img, mask, model_name, heatmap_folder):
        # mask = (np.max(mask) - np.min(mask)) / (mask - np.min(mask))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imshow("cam", np.uint8(255 * cam))
        cv2.imshow("heatmap", np.uint8(heatmap * 255))
        cv2.waitKey()

        # 현재 시간을 이용하여 파일 이름 생성
        current_time = datetime.now().strftime("%Y%m%d%H%M%S_%f")[:-3]
        cam_file_name = f"{current_time}_{model_name}_cam_image.png"
        heatmap_file_name = f"{current_time}_{model_name}_heatmap_image.png"

        # 이미지를 파일로 저장
        cv2.imwrite(os.path.join(heatmap_folder, cam_file_name), np.uint8(255 * cam))
        cv2.imwrite(os.path.join(heatmap_folder, heatmap_file_name), np.uint8(heatmap * 255))

    # model_dir = "Result/train_result/"
    model_dir = "Result/train_result/Kaggle_vgg11_original/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target_class = "edge"
    input_dir = './packaging/test/' + target_class

    # "heatmap" 폴더가 없으면 생성
    heatmap_folder = "Result/heatmap/" + target_class
    os.makedirs(heatmap_folder, exist_ok=True)


    #model_names: "Kaggle_vgg11", "Kaggle_googlenet", "Kaggle_squeezenet", "Kaggle_wideresnet"
    model_name = "Kaggle_vgg11"
    num_classes = 3

    if model_name == "Kaggle_vgg11":
        model_dir = model_dir + model_name
        model = models.vgg11_bn(pretrained=True)
        num_ftrs = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='features', layer='28')

    elif model_name == "Kaggle_googlenet":
        model_dir = model_dir + model_name
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='inception5a', layer='branch4')

    elif model_name == "Kaggle_squeezenet":
        model_dir = model_dir + model_name
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='features', layer='12')

    elif model_name == "Kaggle_wideresnet":
        model_dir = model_dir + model_name
        model = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='layer4', layer='2')


    img_list = os.listdir(input_dir)
    img_list = sorted(glob.glob(os.path.join(input_dir, '*.png')))
    for img_path in img_list:
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        mask = grad_cam(input.cuda(), None)
        show_cam_on_image(img, mask, model_name, heatmap_folder)