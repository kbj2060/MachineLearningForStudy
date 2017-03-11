Tensorflow Chapter 4. 모델 불러오기
==========================

 이번 강의에서는 트레이닝한 모델을 불러들여 weight를 더 트레이닝하거나 새로운 데이터를 테스트할 수 있다. 강의에서 후반부로 갈수록 내가 이해하기엔 좀 힘들어서 모델 불러들이는 부분만 정리해야겠다.

이번엔 test.py와 train.py로 나뉘어서 train.py에서 모델을 트레이닝하고 weight를 model.ckpt에 저장하고 test.py에서 weight를 불러들여 바로 y에 넣는다. 

이 두개를 비교하기 위해 restore.py를 만들었다. 트레이닝한 모델을 불러들여 accuracy를 측정하는 코드이다.

아래 코드는 test.py와 train.py를 diff 명령어를 통해 비교한 것이다. 
```bash
3,4c3,4 # test 코드이기 때문에 데이터를 바꾸었다.
< input_data = [[10,5,7,3,9,11],
< 			[10,5,11,1,21,2]]
---
> input_data = [[1,3,5,3,8,3],
> 			[15,2,14,15,23,25]]
7a8
> 
34a36,41 # 모델을 트레이닝하는 코드들은 모두 지웠다.
> cost = tf.reduce_mean(-y_ * tf.log(y) - (1-y)*tf.log(1-y))
> train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)
> 
> comp_pred = tf.equal(tf.arg_max(y,1) , tf.arg_max(y_, 1 ))
> accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype= tf.float32))
> 
36,38c43,52 # 변수들을 init하는 함수들을 지우고 saver.restore을 이용해 트레이닝된 변수들을 불러들인다.
< saver.restore(sess, 'model.ckpt')
< res = sess.run( y, feed_dict )
< print(res)
---
> init = tf.initialize_all_variables()
> sess.run(init)
> for step in range(1000):
> 	_, loss, acc = sess.run([train, cost, accuracy], feed_dict = feed_dict)
> 	if step % 100 == 0:
> 		saver.save(sess, 'model.ckpt' )
> 		print("step : ",step)
> 		print("loss : ",loss)
> 		print("acc : ",acc)
```

이제 test.py에는 트레이닝 모델은y값만 출력하는 데에 그쳤지만 restore.py를 보면 트레이닝 모델을 통해 label을 추측하는 코드를 볼 수 있다.

```python
import tensorflow as tf

input_data = [[1,3,5,3,8,3],
			[15,2,14,15,23,25]]
label = [[1,0,0,0,0],
		[0,0,1,0,0]]


INPUT_SIZE = 6
HIDDEN1_SIZE = 10
HIDDEN2_SIZE = 8
CLASSES = 5
Learning_Rate = 0.05

x = tf.placeholder(tf.float32, shape=[None,INPUT_SIZE])
y_ = tf.placeholder(tf.float32, shape=[None,CLASSES])

feed_dict = {x:input_data, y_:label}

w1 = tf.Variable( tf.truncated_normal(shape=[INPUT_SIZE, HIDDEN1_SIZE], dtype = tf.float32 ), name='w1' )
b1 = tf.Variable( tf.zeros([HIDDEN1_SIZE], dtype = tf.float32), name='b1' )

w2 = tf.Variable( tf.truncated_normal(shape=[HIDDEN1_SIZE, HIDDEN2_SIZE], dtype = tf.float32 ) ,name='w2')
b2 = tf.Variable( tf.zeros([HIDDEN2_SIZE], dtype = tf.float32), dtype = tf.float32, name='b2' )

wo = tf.Variable( tf.truncated_normal(shape=[HIDDEN2_SIZE,CLASSES], dtype = tf.float32 ), name='wo' )
bo = tf.Variable( tf.zeros([CLASSES], dtype = tf.float32 ), name='bo')

param_list = [w1, b1, w2, b2, wo, bo]
saver = tf.train.Saver(param_list)

hidden1 = tf.sigmoid(tf.matmul(x, w1) + b1)
hidden2 = tf.sigmoid(tf.matmul(hidden1, w2) + b2)
y = tf.sigmoid( tf.matmul( hidden2, wo ) + bo )

cost = tf.reduce_mean(-y_ * tf.log(y) - (1-y)*tf.log(1-y))
train = tf.train.GradientDescentOptimizer(Learning_Rate).minimize(cost)

comp_pred = tf.equal(tf.arg_max(y,1) , tf.arg_max(y_, 1 ))
accuracy = tf.reduce_mean(tf.cast(comp_pred, dtype= tf.float32))

sess = tf.Session()
saver.restore(sess, 'model.ckpt')
for step in range(1000):
	_, loss, acc = sess.run([train, cost, accuracy], feed_dict = feed_dict)
	if step % 100 == 0:
		print("step : ",step)
		print("loss : ",loss)
		print("acc : ",acc)
```
   
   train.py보다 훨씬 코드가 간단해진다. 트레이닝을 하지 않아도 되기 때문에 시간도 훨씬 단축된다. 모델을 불러오는 법을 많이 안쓸거라고 생각하는 사람들도 있겠지만 그렇지 않다. 왜냐하면 머리 좋은 사람들이 몇 주간 트레이닝한 좋은 weight들을 우리는 그냥 불러들여서 이용할 수 있기 때문이다.

아래 그림처럼 restore.py를 실행해보면 acc : 1.0부터 시작하는 것을 보고 트레이닝된 모델을 바로 사용하는 것을 볼 수 있다.

![](./restore.png)

