#sample code: https://csm-kr.tistory.com/74

import os
import cv2
import glob
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F

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
        means = np.array([0.20369699, 0.22522233, 0.22994686])
        stds = np.array([0.14865874, 0.17049677, 0.14530116])

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

    def show_cam_on_image(img, mask):
        # mask = (np.max(mask) - np.min(mask)) / (mask - np.min(mask))
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        cv2.imshow("cam", np.uint8(255 * cam))
        cv2.imshow("heatmap", np.uint8(heatmap * 255))
        cv2.waitKey()


    data_dir = "data/package/"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model_names: "Kaggle_vgg11", "Kaggle_googlenet", "Kaggle_squeezenet", "Kaggle_wideresnet"
    model_name = "Kaggle_googlenet"

    if model_name == "Kaggle_vgg11":
        model = models.vgg11_bn(pretrained=True)
        num_ftrs = model.classifier[0].in_features
        model.classifier = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(data_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='features', layer='28')

    elif model_name == "Kaggle_googlenet":
        model = models.googlenet(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(data_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='inception5a', layer='branch4')

    elif model_name == "Kaggle_squeezenet":
        model = models.squeezenet1_1(pretrained=True)
        model.classifier[1] = nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(data_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='features', layer='12')

    elif model_name == "Kaggle_wideresnet":
        model = models.wide_resnet50_2(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model = model.to(device)
        model.load_state_dict(torch.load(os.path.join(data_dir, model_name + '.pth')))
        print(model)
        model.eval()
        grad_cam = GradCam(model=model, module='layer4', layer='2')

    root = './package/val/Normal'
    img_list = os.listdir(root)
    img_list = sorted(glob.glob(os.path.join(root, '*.jpg')))
    for img_path in img_list:
        img = cv2.imread(img_path, 1)
        img = np.float32(cv2.resize(img, (224, 224))) / 255
        input = preprocess_image(img)
        mask = grad_cam(input.cuda(), None)
        show_cam_on_image(img, mask)