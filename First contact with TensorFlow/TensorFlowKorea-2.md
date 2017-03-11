# 변수간의 관계를 모델로

이번에는 선형 회귀분석 모델을 만들어 보겠습니다. cost function 과 Gradient Descent 같은 머신 러닝 학습에 있어서 중요한 컴포넌트을 어떻게 사용하는지 알아보도록 하겠습니다.

선형 회귀분석은 독립변수 x_i, 상수항 b(random term) 와 종속변수(역주: 결과 값, 즉 y) 사이의 관계를 모델화 하는 것으로 두 변수 사이의 관계일 경우 단순 회귀분석이라고 하며 여러개의 변수를 다루는 다중 회귀분석이 있습니다.

이번에는 텐서플로우가 어떻게 동작하는 지 설명하기 위해 y = W * x + b 형태의 간단한 성형 회귀 분석 모델을 만들 것입니다.

좌푯값들을 생성하기 위해 numpy 패키지를 불러오고 1000개의 x,y 데이터를 만듭니다. 이 데이터는 모델을 만들기 위한 학습 데이터로 사용될 것입다


```python
import numpy as np

num_points = 1000
vectors_set = []
for i in range(num_points):
         x1= np.random.normal(0.0, 0.55)
         y1= x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
         vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]
```

이 데이터의 그래프를 그려보기 위해 matplotlib 패키지를 불러오고
그래프를 show 해봅니다.


```python
import matplotlib.pyplot as plt

plt.plot(x_data, y_data, 'ro', label='Original data')
plt.legend()
plt.show()
```

![pic](https://tensorflowkorea.files.wordpress.com/2016/04/image014.png?w=625)

우리는 W 가 0.1 , b 가 0.3 에 근사한 값이라는 것을 알지만 텐서플로우는 모릅니다. 텐서플로우가 학습을 통해 스스로 찾아내야합니다.

# 코스트 함수(Cost Function)와 그래디언트 디센트(Gradient Descent) 알고리즘

```python
import tensorflow as tf

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b
```

위에서 변수들을 정의했고 이제 평균제곱오차를 기반으로 cost function을 만듭니다.


```python
cost = tf.reduce_mean(tf.square(y - y_data))
```

가장 작은 에러 값을 갖는 직선이 예측 모델로 적합한 직선입니다. 에러를 최소화하는 최적화 알고리즘인 gradient descent 가 하는 역할입니다.
이론적으로 보면 그래디언트 디센트는 일련의 파라메타로 된 함수가 주어지면 초기 시작점에서 함수의 값이 최소화 되는 방향으로 파라메타를 변경하는 것을 반복적으로 수행하는 알고리즘입니다. 함수의 기울기가 음의 방향인 쪽으로 진행하면서 반복적으로 최적화를 수행합니다. 보통 양의 값을 만들기 위해 거리 값을 제곱하는 것이고 기울기를 계산해야 하므로 에러 함수는 미분 가능해야 합니다.


```python
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(cost)
```

이제 텐서플로우가 내부 데이터 구조에 관련 데이터를 생성한다는 것을 알아 보았고 코스트함수를 등록하여 그래디언트 디센트 알고리즘 옵티마이저(optimizer) train을 구현했습니다. 학습 속도(learning rate)는 0.5로 지정했습니다.

# 알고리즘 실행

여기까지는 텐서플로우 라이브러리를 호출하는 코드는 단지 내부 그래프 구조에 정보를 추가시킨 것일 뿐 텐서플로우의 실행 모듈은 아직 아무런 알고리즘도 실행하지 않았습니다. 그러므로 이전 챕터에서와 같이 session을 생성하고 run 메소드를 train 파라메타와 함께 호출해야 합니다.  또한 변수를 선언했으므로 아래와 같은 명령으로 초기화해야 합니다.


```python
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
```

전에는 placeholder를 사용할때, run() 메소드에서 feed_dict을 통해 값을 지정해주었다면 이번에는 variable이기 때문에 initialize_all_variables() 를 사용해야한다. 


```python
for step in range(100):
   sess.run(train)
print(step, sess.run(W), sess.run(b))
```

    99 [ 0.09965742] [ 0.29854473]


step 수를 늘릴수록 점점 W는 0.1에 b는 0.3에 근사합니다.

그래프를 통해 보기 위해 아래의 코드를 추가합니다.

```python
plt.plot(x_data, y_data, 'ro')
plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
plt.legend()
plt.show()
```

![pic](https://tensorflowkorea.files.wordpress.com/2016/04/image016.png?w=625)

# Referenece
* http://cs231n.github.io/convolutional-networks/
* https://tensorflowkorea.wordpress.com/5-텐서플로우-다중-레이어-뉴럴-네트워크-first-contact-with-tensorflow/
