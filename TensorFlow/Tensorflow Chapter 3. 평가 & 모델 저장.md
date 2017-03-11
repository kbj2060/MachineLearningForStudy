Tensorflow Chapter 3. 평가 & 모델 저장
===============
1. Evaluation
-------
모델을 평가하기 위해 cost를 측정해 비교할 수 있었다. 그 전에 reduce_sum이라는 함수를 굉장히 많이 쓰이므로 공부하고 넘어간다. RSS를 구하기 위해서도 reduce_sum 함수를 사용한다. 행렬의 원하는 원소의 합을 구할 수 있는 함수이다. 이 강의에서 깨달은 것은 batch에 관한 개념이었다. 이 예시에서는 데이터가 2개 밖에 없어 풀배치로 Train을 했기 때문에 cost_=tf.reduce_sum((-y_*tf.log(y)-(1-y)*tf.log(1-y)), 1) 연산을 해보면 원소는 2개가 나온다. 즉, batch의 갯수와 cost 행렬의 행의 개수는 같다.


역시 함수에 대한 이해는 텐서플로우 공식 다큐먼트를 보고 하는 것이 옳다. 

> 함수 설명
> tf.reduce_sum(input_tensor, axis=None, keep_dims = False, name = None, reduction_indices = None)
> 
> 사용 예시
>  'x' is [[1, 1, 1], [1, 1, 1]]
tf.reduce_sum(x) ==> 6
tf.reduce_sum(x, 0) ==> [2, 2, 2]
tf.reduce_sum(x, 1) ==> [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]
> tf.reduce_sum(x, [0, 1]) ==> 6

그리고 기존 코드에서 아래와 같은 코드를 추가해 accuracy를 확인한다.
```python
comp_pred = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype=tf.float32))
```
y행렬의 각 행에서 가장 큰 값을 가진 값의 인덱스와 y_행렬의 각행에거 가장 큰 값을 가진 값의 인덱스를 비교해 같으면 True, 다르면 False을 갖는 행렬을 comp_pred에 저장한다.
그 이후에 bool 값을 갖는 comp_pred 행렬을 0과 1로 만들고 평균을 내 정확도를 저장한다.


2. Model Save
------

텐서플로우에서는 같은 형식의  데이터셋에서 이미 트레이닝된 모델을 사용하여 예측을 하고 싶을 때 체크포인트라는 개념을 이용한다. 체크포인트에 트레이닝된 weight 정보를 저장하여 이용할 수 있다.

두 줄만 추가하면 체크포인트를 생성할 수 있다. 
```python
param_list = 
[W_h1, b_h1, W_h2, b_h2, w_o, b_o]
saver = tf.train.Saver(param_list)
saver.save(sess, 'model')
```
param_list 는 굳이 추가해주지 않아도 된다. 정확히 추가하기 위해 한 것이다. 나중에 제대로 된 모델을 만든다면 사용하고 싶다.

위 내용을 추가한 예시는 아래와 같다.
```python
import tensorflow as tf

input_data = [[1,5,3,7,8,10,12],
                    [5,8,10,3,9,7,1]]
label_data = [[0,0,0,1,0],
                     [0,0,1,0,0]]

# 좋은 코드의 예
INPUT_SIZE = 7
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5
Learning_Rate = 0.05

# 행렬인 데이터들을 텐서로 만들어주기 위해 placeholder를 이용해 선언
x = tf.placeholder( tf.float32, shape = [None, INPUT_SIZE])
y_ = tf.placeholder( tf.float32, shape = [None,CLASSES] )
feed_dict = {x: input_data, y_: label_data}

# hidden layer의 weight와 bias 선언
W_h1 = tf.Variable( tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE]), dtype=tf.float32)
b_h1 = tf.Variable( tf.zeros([HIDDEN1_SIZE], dtype = tf.float32))

W_h2 = tf.Variable( tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE]), dtype=tf.float32)
b_h2 = tf.Variable( tf.zeros([HIDDEN2_SIZE], dtype = tf.float32))

w_o = tf.Variable( tf.truncated_normal(shape=[HIDDEN2_SIZE, CLASSES]), dtype=tf.float32)
b_o = tf.Variable( tf.zeros([CLASSES], dtype = tf.float32))

# 변수 선언 바로 밑에 saver 선언해 저장할 weight를 넣는다.
param_list = [W_h1, b_h1, W_h2, b_h2, w_o, b_o]
saver = tf.train.Saver(param_list)

# layer안의 계산
hidden1 = tf.sigmoid(tf.matmul( x, W_h1) + b_h1)
hidden2 = tf.sigmoid(tf.matmul( hidden1 , W_h2) + b_h2)
y = tf.sigmoid(tf.matmul( hidden2, w_o) + b_o)

# 정확도를 위한 cost와 정확도를 높이기 위해 train 선언
cost = tf.reduce_mean(-y_*tf.log(y)-(1-y)*tf.log(1-y))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

# accuracy를 체크
comp_pred = tf.equal(tf.arg_max(y, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype=tf.float32))

# 이 때까지 그래프를 그렸다면 이제 세션에 넣고 계산
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)
for i in range(1000):
    _, loss,acc = sess.run([train,cost, accuracy] , feed_dict = feed_dict)
    if i % 100 == 0:
        saver.save(sess, 'model.ckpt')
        print("step : ",i)
        print("loss : ",loss)
        print("acc : ",acc) 
```















