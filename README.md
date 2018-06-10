# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/barchart.png "class label"
[image2]: ./examples/preprocessing.png "Preprocessing"
[image3]: ./examples/real_sign.png "Real Images"
[image4]: ./examples/top5.png "Softmax"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/bt22dr/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization

Here is an exploratory visualization of the data set. 아래 그림은 training/validation/test dataset의 클래스 label 분포를 보여준다. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing 

normalization과 grayscale 변환 후 데이터셋을 학습했을 때 validation accuracy가 높게 나왔기 때문에 이 두 가지 방법으로 전처리를 수행하였고 별도의 data augmentation은 수행하지 않았다. `[:,:,None]`과 같이 처리를 해주면 (32,32) 차원 이미지를 (32,32,1) 차원으로 바뀌기 때문에 차후 별도의 expand_dims() 처리를 해주지 않아도 된다. 

``` python
dataset = (dataset - 128.0) / 128.0
...
img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:,:,None] 
```

![alt text][image2]

#### 2. Model architecture 

My final model consisted of the following layers:

| Layer                 |     Description                               | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 32x32x1 gray scale image                      |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6    |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 14x14x6                  |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU                  |                                               |
| Max pooling           | 2x2 stride,  outputs 5x5x16                   |
| Flatten               | input = 5x5x16, output = 400                  |
| Fully connected       | input = 400, output = 120                     |
| RELU                  |                                               |
| DROPOUT               |                                               |
| Fully connected       | input = 120, output = 84                      |
| RELU                  |                                               |
| DROPOUT               |                                               |
| Fully connected       | input = 84, output = 10                       |
| Softmax               |                                               |


#### 3. Hyperparameters

모델을 학습하기 위해 아래와 같은 하이퍼파라미터를 사용하였다. 

* learning_rate = 0.001, 
* L2 regularzation = 0.005
* AdamOptimizer
* number of epochs = 30
* batch size = 128

#### 4. Approach

처음에는 단순히 LeNet 아키텍처를 사용했는데 overfitting이 강하게 발생하여 dropout과 l2 regularization을 추가하였고 최종적인 성능은 아래와 같다. 

* training set accuracy of 0.999
* validation set accuracy of 0.946
* test set accuracy of 0.932

### Test a Model on New Images

#### 1. New Images

Here are five German traffic signs that I found on the other github [repository](https://github.com/frankkanis/CarND-Traffic-Sign-Classifier-Project/tree/master/new_signs):

![alt text][image3]

#### 2. Discuss the model's predictions

Here are the results of the prediction:

| Image			        |     Prediction	    	| 
|:---------------------:|:-------------------------:| 
| speed_limit_30  		| speed_limit_70        	| 
| yield 		        | yield 			       	|
| stop 		            | stop				     	|
| no_entry   	      	| no_entry			       	|
| keep_right 	    	| keep_rightd          	 	|

The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93%.

#### 3. Top 5 softmax probabilities

아래 그림을 보면 대부분의 이미지에 대해 100%에 가까운 확률로 예측한 것을 알 수 있다. 

![alt text][image4]
