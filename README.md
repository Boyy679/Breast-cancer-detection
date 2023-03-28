# Breast-cancer-detection
#### 1. Algorithm use

This is a compute vision project mainly use the following algorithms：

* typical machine learning method
  * Logical Regression
  * Random Forest
  * GNB
  * DTC
  * KNN
  * SVM
* Self build CNN model
* transfer learning model
  * DenseNet121
  * DenseNet161
  * DenseNet169
  * DenseNet201
  * MobileNet
  * MobileNetV2
  * Resnet50
  * VGG16
  * VGG19

#### 2. Data introduction

> **Dataset** 
>
> ​	X.npy: contain the images
>
> ​	Y.npy: contain the labels

To classify 2 kinds breast cancer images：

* **0**: breast cancer negative (IDC-)
* **1**: breast cancer positive (IDC+)

#### 3.  Result analysis

##### 3.1 machine learning algorithm

For the machine learning model:

|  **Model**   |    LR    |    RF    |   KNN    |   SVM    |   GNB    |   DTC    |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------: |
| **Accuracy** | 0.709009 | 0.763964 | 0.702703 | 0.773874 | 0.708108 | 0.679279 |
|   **Loss**   | 0.043027 | 0.042790 | 0.055095 | 0.044583 | 0.025861 | 0.025225 |

From the table we see the SVM has the most accuracy, which is **0.773874  **, and DTC has least loss, which is **0.025225**.

##### 3.2 CNN model

The CNN model has **0.9387** accuracy, **0.7568** validation accuracy, **0.1576** loss.

##### 3.3 Transfer learning model

For the transfer learning model：

|    Model    | Index  |          |           |              |      |      |
| :---------: | :----: | :------: | :-------: | :----------: | :--: | :--: |
|   IDC(-)    | IDC(+) | Accuracy | Macro avg | Weighted avg |      |      |
|  Precision  | VGG19  |   0.64   |    0.8    |      -       | 0.72 | 0.72 |
| DenseNet201 |  0.76  |   0.78   |     -     |     0.77     | 0.77 |      |
|   Recall    | VGG19  |   0.52   |   0.87    |      -       | 0.7  | 0.69 |
| DenseNet201 |  0.79  |   0.75   |     -     |     0.77     | 0.77 |      |
|  F1-score   | VGG19  |   0.57   |   0.83    |     0.7      | 0.7  | 0.7  |
| DenseNet201 |  0.77  |   0.76   |   0.77    |     0.77     | 0.77 |      |
|   support   | VGG19  |   558    |    552    |     1110     | 1110 | 1110 |
| DenseNet201 |  556   |   554    |   1110    |     1110     | 1110 |      |











