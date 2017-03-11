
Tensorflow Chapter 2. 모델 설계하기
==============================================

 이번 강의에서는 간단한 모델을 만들어 실행해보는 예제 구현을 한다. 머신 러닝을 조금 공부해본 사람들이 이해하기 쉬운 강의였다고 생각한다. 편집본을 봐서 그런지 weight, bias 에 대한 지식은 안다고 가정하고 강의하시는 것 같다. weight나 bias 같은 개념을 조금이나마 알기 때문에 weight과 bias 값에는 tf.Variable을 통해 값을 저장한다는 것을 이해할 수 있었다. 만약 모르는 사람이라면 constant나 placeholder로 선언하지 왜 굳이 variable인가 라고 의문을 던질 수 있다고 생각한다.

------
우선 강의에서 사용한 예제를 살펴보자면 아래와 같다.

```python
import tensorflow as tf
import numpy as np

input_data = [[1,5,3,7,8,10,12],
                    [5,8,10,3,9,7,1]]
label_data = [[0,0,0,1,0],
                     [1,0,0,0,0]]

# 좋은 코드의 예
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5
Learning_Rate = 0.01

# 행렬인 데이터들을 텐서로 만들어주기 위해 placeholder를 이용해 선언
x = tf.placeholder( tf.float32, shape = [None, INPUT_SIZE])
y_ = tf.placeholder( tf.float32, shape = [None,CLASSES] )언
feed_dict = {x: input_data, y_: label_data}

# hidden layer의 weight와 bias 선언
W_h1 = tf.Variable( tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32)
b_h1 = tf.Variable( tf.zeros([HIDDEN1_SIZE], dtype = tf.float32))

W_h2 = tf.Variable( tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
b_h2 = tf.Variable( tf.zeros([HIDDEN2_SIZE], dtype = tf.float32))

w_o = tf.Variable( tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)
b_o = tf.Variable( tf.zeros([CLASSES], dtype = tf.float32))

# layer안의 계산
hidden1 = tf.sigmoid(tf.matmul( x, W_h1) + b_h1)
hidden2 = tf.sigmoid(tf.matmul( hidden1 , W_h2) + b_h2)
y = tf.sigmoid(tf.matmul( hidden2, w_o) + b_o)

# 정확도를 위한 cost와 정확도를 높이기 위해 train 선언
cost = tf.reduce_mean(-y_*tf.log(y)-(1-y)*tf.log(1-y))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

# 이 때까지 그래프를 그렸다면 이제 세션에 넣고 계산
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
for i in range(100):
    _, loss = sess.run([train,cost] , feed_dict = feed_dict)
    print("step : ",i)
    print("loss : ",loss)

#세션 종료
sess.close()
```
매우 간단한 모델이다. 모델과 데이터가 맞지 않지만 우선 코드의 흐름을 아는 것이 중요하다. (모델은 linear 모델인데 데이터가 classification 데이터인 것 같다.) 가장 베이직한 모델이기 때문에 이 모델을 응용하면 논문에 사용된 모델도 따라할 수 있으리라 본다.  무작정 기본 모델을 공부하는 경우와 텐서플로우에 대한 이해를 높인 후에 보는 것은 하늘과 땅 차이라고 생각한다. 지금 내가 그렇다.

> 위 코드의 결과

> step :  0
loss :  0.4208
step :  1
loss :  0.420211
step :  2
loss :  0.419622
step :  3
loss :  0.419034
step :  4
loss :  0.418446

>...

> step :  95
loss :  0.370048
step :  96
loss :  0.369587
step :  97
loss :  0.369126
step :  98
loss :  0.368667
step :  99
> loss :  0.36821

-----------------------------------------------------------------
결과를 보면 알 수 있듯이 loss가 줄어드는 것을 볼 수 있다. 딥러닝을 하기 전에 텐서플로우에 대한 이해를 높이는 것을 강력히 추천한다. 물론 다른 라이브러리를 사용한다면 상관 없겠지만 텐서플로우는 배울수록 매력있는 라이브러리라고 생각한다. 위와 같은 기본 모델은 다른 글에도 많이 썼기 때문에 생략한다.

















