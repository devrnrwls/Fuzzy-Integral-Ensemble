# Fuzzy-Integral-Ensemble
This is the official implementation of the paper titled "[Fuzzy Integral-based CNN Ensemble for COVID-19 Detection from Lung CT Images](https://doi.org/10.1016/j.compbiomed.2021.104895)" published in _Computers in Biology and Medicine_, Elsevier.

## Requirements

To install the dependencies, run the following using the command prompt:

`pip install -r requirements.txt`

## Running the code on the COVID data

Download the dataset from [Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset) and split it into train and validation sets in 80-20 ratio. (validation(20) => covid: 376, non-COVID: 369)

Required Directory Structure:
```

.
+-- data
|   +-- .
|   +-- train
|   +--      |+-- COVID
|   +--      |+-- non-COVID
|   +-- val
|   +--      |+-- COVID
|   +--      |+-- non-COVID
+-- main.py
+-- probability_extraction.py
+-- ensemble.py
+-- sugeno_integral.py

```

# Run: 
```
python main.py --data_directory ./data/ --epochs 100`
```

# Result
```
Epoch 25/25
----------
train Loss: 0.0031 Acc: 1.0000
val Loss: 0.7461 Acc: 0.7718

Training complete in 6m 19s
Best val Acc: 0.837584

Getting the Probability Distribution
Accuracy =  83.75838926174497
Accuracy =  0.8389261744966443
Classification Report:
              precision    recall  f1-score   support

       COVID     0.8245    0.8516    0.8378       364
   Non-COVID     0.8537    0.8268    0.8400       381

    accuracy                         0.8389       745
   macro avg     0.8391    0.8392    0.8389       745
weighted avg     0.8394    0.8389    0.8389       745

Confusion matrix:
[[310  54]
 [ 66 315]]

Classwise Accuracy :[0.85164835 0.82677165]

Balanced Accuracy Score:  0.8392100025958293

Process finished with exit code 0

```

# 기타
torch vision에서 이용 가능한 models 확인: https://github.com/pytorch/vision/tree/main/torchvision/models

(처음에는 main 브랜치로 되어 있으니 사용중인 버전의 브랜치로 이동 후 "__init__.py" 에서 모델 확인)

# Citation
If you found this repository helpful, please consider citing our paper:
```
@article{kundu2021covid,
  title={COVID-19 detection from lung CT-Scans using a fuzzy integral-based CNN ensemble},
  author={Kundu, Rohit and Singh, Pawan Kumar and Mirjalili, Seyedali and Sarkar, Ram},
  journal={Computers in Biology and Medicine},
  volume={138},
  pages={104895},
  year={2021},
  publisher={Elsevier}
}
```
