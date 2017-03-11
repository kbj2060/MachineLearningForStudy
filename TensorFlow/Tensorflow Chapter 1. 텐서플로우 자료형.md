Tensorflow Chapter 1. 텐서플로우 자료형
======================================

Youtube에서 chanwoo lee님께서 올린 [동영상](https://www.youtube.com/watch?v=OJH-KnUK9JY&t=2912s)을 정리하고자 쓴 글이다.
그냥 딥러닝 공부하다가 쓰이는 라이브러리라고 생각하고 공부했다고 좌절한 기억이 있다. 그래서 텐서플로우부터 공부하고자 한다. 이찬우님께서 올리신 동영상을 봤는데 굉장히 설명이 자세하고 텐서플로우 라이브러리에 대한 이해를 높일 수 있는 기회가 될 수 있겠다고 생각했다.

첫 수업은 크게 2가지로 나뉜다.
* 텐서플로우의 기본 작동 원리
* 자료형에 대한 이해

1. 텐서플로우의 기본 작동 원리
-----------------------------------------------
텐서플로우의 작동 원리를 크게 보자면 session이라는 큰 틀에 그래프를 넣고 session을 device(ex.gpu,cpu)에서 실행시키는 것이다. 간단히 예를 들어보자면,  

```python
import tensorflow as tf

ph = tf.placeholder(dtype=tf.float32)
const = tf.constant([3])
const1 = tf.constant([4])
var = tf.Variable([2])

result_const = const + const1

sess = tf.Session() //session을 열고
res = sess.run(result_const) //result_const안에 있는 그래프를 session에 넣고 session을 device에서 연산하고 값을 반환
print(res)
```

2. 텐서플로우 자료형에 대한 이해
-------------------------------------------------

### 2.1 constatnt
: constatnt 위 예제와 같이 선언되며 초기화됨과 동시에 노드 안에 값을 갖게 된다. 그래서 따로 초기화를 안해주어도 되며 result_const값을 run안에 넣어주면 연산하게 된다.
### 2.2 variable
: variable은 constant와는 조금 다르다. constant은 초기화 되고 바로 노드에 값이 저장되지만 variable은 빈 노드를 그래프에 그려넣어 놓고 tf.initialize_all_varialbes() 함수를 통해 값을 넣게 된다.

```python
import tensorflow as tf

var = tf.Variable([3])
var1 = tf.Variable([2])

res_var = var + var1

sess = tf.Session()
#res = sess.run(res_var) //이렇게 하면 에러가 뜬다.
init = tf.initialize_all_variables()
sess.run(init) #init 을 실행해주어야 노드 안에 값을 넣을 수 있다.
res = sess.run(res_var)
print(res)
```
### 2.3 placeholder
: placeholder는 초기화 부분에 값을 넣지 않아도 된다. 텐서가 아닌 것들을 텐서가 될 수 있도록 맵핑시켜주는 역할을 한다. 왜냐하면 텐서는 텐서와 연산이 가능한데 텐서가 아닌 행렬이 텐서와 연산하려면 이 기능을 사용해야 한다. 맵핑하는 데에 feeding이 필요하다.

```python
import tensorflow as tf

ph1 = tf.placeholder(dtype=tf.float32)
ph2 = tf.placeholder(dtype=tf.float32)

mat = [1,2,3,4,5]
flag = [10, 20, 30, 40, 50]

result_ph = ph1 + ph2
feed_dict = {ph1: mat, ph2: flag}

sess = tf.Session() #session을 열고
res = sess.run(result_ph, feed_dict= feed_dict) #result_const안에 있는 그래프를 session에 넣고 session을 device에서 연산하고 값을 반환
print(res)
```