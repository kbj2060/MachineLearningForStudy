
선형 회귀분석은 모델을 만들기 위해 입력 데이터와 출력 값(혹은 레이블, label)을 사용했다는 점에서 감독(supervised) 학습 알고리즘입니다.하지만 모든 데이터에 레이블이 있는 것은 아닙니다. 

이럴 때 클러스터링(clustering, 군집화)이라는 비감독(unsupervised) 학습 알고리즘을 사용할 수 있습니다. 클러스터링은 데이터 분석의 사전 작업으로 사용되기 좋아 널리 이용되는 방법입니다.

K-means 클러스터링 알고리즘을 소개합니다. K-means 알고리즘은 데이터를 다른 묶음과 구분되도록 유사한 것끼리 자동으로 그룹핑해 주기 때문에 가장 많이 알려졌고 널리 사용됩니다. 이 알로리즘에서는 예측해야 할 타겟 변수나 결과 변수가 없습니다.

텐서플로우에 대해 더 알기 위해 텐서(tensor)라 불리는 기본 데이터 구조에 대해 자세히 살펴 보겠습니다. 텐서 데이터가 어떤 것인지 먼서 설명하고 제공되는 기능에 대해 소개하겠습니다. 그리고 나서 텐서를 이용하여 K-means 알고리즘을 예제로 풀어보도록 하겠습니다.

# 기본 데이터 구조 : 텐서 (tensor)

텐서플로우는 텐서라는 기본 데이터 구조로 모든 데이터들을 표현합니다. 텐서는 동적 사이즈를 갖는 다차원 데이터 배열로 볼 수 있으며 불리언(boolean), 문자열(string)이나 여러 종류의 숫자형 같은 정적 데이터 타입을 가집니다.

그리고 각 텐서는 배열의 차원을 나타내는 랭크(rank)를 가집니다. 예를 들면 다음의 텐서(파이썬에서 list로 표현된)는 랭크 2를 가집니다.

t = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]

텐서의 랭크에는 제한이 없습니다. 랭크 2인 텐서는 행렬을 나타내며 랭크 1인 텐서는 벡터로 볼 수 있습니다. 랭크 0 텐서는 스칼라(scalar) 값이 됩니다.

텐서플로우의 공식 문서에서는 텐서의 차원을 표현하기 위해 구조(Shape), 랭크(Rank), 차원번호(Dimension Number) 라는 세가지 종류의 이름을 사용합니다. 아래 테이블은 텐서플로우 문서를 볼 때 혼돈되지 않도록 각 이름 사이의 관계를 나타냈습니다.

|구조(SHAPE)	|랭크(RANK)	|차원번호(DIMENSION NUMBER) |
|---------------|-----------|---------------------------|
|[]	            |0	        |0-D                        |
|[D0]	        |1	        |1-D                        |
|[D0, D1]	    |2	        |2-D                        |
|[D0, D1, D2]	|3	        |3-D                        |
|…	            |…	        |…                          |
|[D0, D1, … Dn]	|n	        |n-D                        |

이 장을 진행하면서 아래 중 일부 함수에 대해 자세히 설명하겠습니다. 변환 함수의 전체 목록과 설명은 텐서플로우 공식 웹사이트의 텐서 변환(Tensor Transformations)에서 찾을 수 있습니다.


|함수	        |설명                                                                   |
|---------------|-----------------------------------------------------------------------|
|tf.shape	    |텐서의 구조를 알아냅니다.                                              |
|tf.size	    |텐서의 크기를 알아냅니다.                                              |
|tf.rank	    |텐서의 랭크를 알아냅니다.                                              |
|tf.reshape	    |텐서의 엘리먼트(element)는 그대로 유지하면서 텐서의 구조를 바꿉니다.   |
|tf.squeeze	    |텐서에서 크기가 1인 차원을 삭제합니다.                                 |
|tf.expand_dims	|텐서에 차원을 추가합니다.                                              |
|tf.slice	    |텐서의 일부분을 삭제합니다.|
|tf.split	    |텐서를 한 차원을 기준으로 여러개의 텐서로 나눕니다.|
|tf.tile	    |한 텐서를 여러번 중복으로 늘려 새 텐서를 만듭니다.|
|tf.concat	    |한 차원을 기준으로 텐서를 이어 붙입니다.|
|tf.reverse	    |텐서의 지정된 차원을 역전시킵니다.|
|tf.transpose	|텐서를 전치(transpose)시킵니다.|
|tf.gather	    |주어진 인덱스에 따라 텐서의 엘리먼트를 모읍니다.|

예를 들어, 2×2000 배열(2D 텐서)을 3차원 배열(3D 텐서)로 확장하고 싶다면 tf.expand_dims 함수를 사용하여 텐서의 원하는 위치에 차원을 추가할 수 있습니다.

```python
vectors = tf.constant(conjunto_puntos)
extended_vectors = tf.expand_dims(vectors, 0)
```

여기서 tf.expand_dims 은 파라메타로 지정된 텐서의 위치(0부터 가능)에 하나의 차원을 추가하였습니다.(역주: 2차원 텐서의 경우 지정할 수 있는 차원은 0, 1 입니다)

위 변환 과정을 그림으로 보면 아래와 같습니다.

![image](https://tensorflowkorea.files.wordpress.com/2016/05/image023.gif?w=230&h=422)

그림에서 볼 수 있듯이 우리는 이제 3D 텐서를 얻었습니다. 하지만 함수의 인자로 전달하는 새로운 차원인 D0에 크기를 지정할 수는 없습니다.

get_shape() 함수로 이 텐서의 크기를 확인하면 D0에는 크기가 없다는 걸 알 수 있습니다.

```python
print expanded_vectors.get_shape()
```

아래와 같은 결과를 얻습니다.

TensorShape([Dimension(1), Dimension(2000), Dimension(2)])

이 장의 후반부에 보게 되겠지만 텐서플로우의 텐서 구조 전파(shape broadcasting) 기능 덕에 텐서를 다루는 많은 수학 함수들(1장에서 소개된)은 사이즈가 없는 차원을 스스로 인식하고 가능한 값을 추측하여 사이즈를 결정하게 합니다.

# K-means 알고리즘

K-means는 클러스터링 문제를 풀기위한 비감독 알고리즘입니다. 이 알고리즘은 간단한 방법으로 주어진 데이터를 지정된 클러스터 갯수(k)로 그룹핑합니다. 한 클러스터 내의 데이터들은 동일한 성질을 가지며 다른 그룹에 대하여 구별됩니다. 즉 한 클러스터 내의 모든 엘리먼트들은 클러스터 밖의 데이터 보다 서로 더 닮아 있습니다.

알고리즘의 결과는 센트로이드(centroid)라 불리는 K개의 포인트로서 서로 다른 그룹의 중심을 나타내며 데이터들은 K 클러스터 중 하나에만 속할 수 있습니다. 한 클러스터 내의 모든 데이터들은 다른 센트로이드 보다 자신의 센트로이드와의 거리가 더 가깝습니다.

클러스터를 구성하는 데 직접 에러 함수를 최소화하려면 계산 비용이 매우 많이 듭니다.(NP-hard 문제로 알려져 있음) 그래서 스스로 로컬 최소값에 빠르게 수렴할 수 있는 방법들이 개발되어 왔습니다. 가장 널리 사용되는 방법은 몇번의 반복으로 수렴이 되는 반복 개선(iterative refinement) 테크닉 입니다.

대체적으로 이 테크닉은 세개의 단계를 가집니다.

    초기 단계(step 0): K 센트로이드의 초기 값을 결정한다.
    할당 단계(step 1): 가까운 클러스터에 데이터를 할당한다.
    수정 단계(step 2): 각 클러스터에 대해 새로운 센트로이드를 계산한다.

K개 센트로이드의 초기 값을 정하는 방법에는 몇가지가 있습니다. 그 중 하나는 데이터 중 K 개를 임의로 선택하여 센트로이드로 삼는 것 입니다. 우리 예제에서는 이 방법을 사용하겠습니다.

할당 단계와 수정 단계는 알고리즘이 수렴되어서 클러스터 내의 데이터의 변화가 없을 때 까지 루프 안에서 반복됩니다.

이 알고리즘은 휴리스틱한 방법이므로 진정한 최적값으로 수렴한다는 보장은 없으며 결과는 초기 센트로이드를 어떻게 정했는지에 영향을 받습니다. 일반적으로 이 알고리즘의 속도는 매우 빠르므로 초기 센트로이드를 바꿔 가면서 여러번 알고리즘을 수행하여 최종 결과를 만들도록 합니다.

텐서플로우에서 K-means 예제를 시작하기 위해 먼저 샘플 데이터를 생성합니다. 결과 값을 그럴싸하게 만들기 위해 두개의 정규분포를 이용하여 2D 좌표계에 2000개의 포인트를 난수로 발생시킵니다. 아래 코드를 참고하세요.


```python
import tensorflow as tf
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
   if np.random.random() > 0.5:
     conjunto_puntos.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
   else:
     conjunto_puntos.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])
        
df = pd.DataFrame({"x": [v[0] for v in conjunto_puntos],"y": [v[1] for v in conjunto_puntos]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()
```

![pic](https://i1.wp.com/www.jorditorres.org/wp-content/uploads/2016/02/image024.png)

텐서플로우에서 위 데이터를 4개의 클러스터로 그룹핑하는 K-means 알고리즘 구현 코드는 아래와 같습니다.(Shawn Simister의 블로그에 올라온 모델을 참고했습니다)

```python
vectors = tf.constant(conjunto_puntos)
k = 4
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)

means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)])

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(100):
   _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
```

위 코드를 하나씩 분석해보겠습니다. 
첫번째로 할 일은 샘플 데이터를 텐서로 바꾸는 일입니다.
텐서플로우 프로그래밍은 텐서를 이용한 프로그래밍임을 잊어서는 안됩니다. 그러므로 numpy로 할당된 변수도 텐서로 바꾸어줘야 합니다.

위에서 언급했듯이 무작위로 k개의 데이터를 선택된 센트로이드를 할당해야합니다. 코드에서는 centroides 변수입니다. centroides 변수에는 k개의 무작위로 선택된 좌표가 저장 되어 있습니다.
아래와 같이 텐서의 크기를 알 수 있습니다.

```python
print vectors.get_shape()
print centroides.get_shape()

TensorShape([Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2)])
```

vectors는 D0 차원에 2000개의 배열을 가지고 있고 D1 차원에는 각 포인트의 x, y 좌표의 값을 가지고 있습니다. 반면 centroids는 D0 차원에 4개, D1 차원에 vectors와 동일한 2개의 배열을 가진 행렬입니다.

다음 알고리즘은 루프 반복 부분입니다. 먼저 각 포인트에 대해 유클리디안 제곱거리(Squared Euclidean Distance)로(거리를 비교할 때 사용할 수 있음) 가장 가까운 센트로이드를 계산합니다.

이 값을 계산하기 위해 tf.sub(vectors, centroids)를 사용합니다. 주의할 점은 뺄셈을 하려고 하는 두 텐서가 모두 2차원이지만 1차원 배열의 갯수가(D0 차원이 2000 vs 4) 다르다는 것 입니다.
이 문제를 해결하기 위해 이전에 언급했던 tf.expand_dims 함수를 사용하여 두 텐서에 차원을 추가합니다. 이렇게 하는 이유는 두 텐서를 2차원에서 3차원으로 만들어 뺄셈을 할 수 있도록 사이즈를 맞추려는 것 입니다

```python
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)
```

tf.expand_dims 은 두 텐서에 각각 하나의 차원을 추가합니다. vectors 텐서에는 첫번째 차원(D0)를 추가하고 centroids 텐서에는 두번째 차원(D1)을 추가합니다. 그림으로 보면 각 차원들은 확장된 텐서에서도 동일한 의미를 가지고 있습니다.

![pic](https://i2.wp.com/www.jorditorres.org/wp-content/uploads/2016/02/image031.gif)

하지만 추가한 차원의 크기가 모두 1로 아직 결정되지 않았다는 것을 의미합니다. 전에 언급한 텐서플로우 broadcasting 기능이 두 텐서의 엘리먼트를 어떻게 빼야할 지 스스로 알아낼 수 있습니다. 정말 강력한 라이브러리인 것 같습니다.

위 그림을 보면 두 텐서의 구조가 같은 부분 즉 어떤 차원이 같은 크기인지 알아채어 차원 D2 에서 뺄셈이 됩니다. 대신 차원 D0는 expanded_centroids 에서만 크기가 정해져 있습니다.

이런 경우 텐서플로우는 expanded_vectors 텐서의 D0 차원의 크기가 expanded_centroids의 D0 차원의 크기와 같다고 가정합니다. 그래서 각 엘리먼트 별로 뺄셈이 이루어지게 됩니다.

그리고 expanded_centroids 텐서의 D1 차원에서도 같은 일이 벌어집니다. 즉 텐서플로우는 expanded_vectors 텐서의 D1 차원과 같은 사이즈로 간주합니다.

유클리디안 제곱거리(Squared Euclidean Distance)를 사용하는 할당 단계(step 1)의 알고리즘은 텐서플로우에서 4줄의 코드로 나타낼 수 있습니다.

```python
diff=tf.sub(expanded_vectors, expanded_centroides)
sqr= tf.square(diff)
distances = tf.reduce_sum(sqr, 2)
assignments = tf.argmin(distances, 0)
```

diff, sqr, distance, assignment 텐서의 크기를 살펴보면 아래와 같습니다.

TensorShape([Dimension(4), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2000), Dimension(2)])
TensorShape([Dimension(4), Dimension(2000)])
TensorShape([Dimension(2000)])

tf.sub 함수는 expaned_vectors 와 expanded_centroids 를 뺀 값을 가진 diff 텐서를 리턴합니다.(D0 차원에는 센트로이드, D1 차원에는 데이터 인덱스, D2 차원에는 x, y 값을 가진 텐서)

sqr 텐서는 diff 텐서의 제곱 값을 가집니다. distances 텐서에서는 tf.reduce_sum 메소드에 파라메타로 지정된 차원(D2)가 감소된 것을 볼 수 있습니다.

텐서플로우는 tf.reduce_sum 처럼 텐서의 차원을 감소시키는 수학 연산을 여럿 제공하고 있습니다. 아래 테이블에 중요한 몇개를 요약했습니다


|함수	|설명|
|-------|-------------------------------------------------|
|tf.reduce_sum	|지정된 차원을 따라 엘리먼트들을 더합니다.|
|tf.reduce_prod	|지정된 차원을 따라 엘리먼트들을 곱합니다.|
|tf.reduce_min	|지정된 차원을 따라 최소값을 선택합니다.|
|tf.reduce_max	|지정된 차원을 따라 최대값을 선택합니다.|
|tf.reduce_mean	|지정된 차원을 따라 평균값을 계산합니다.|

마지막으로 센트로이드의 선택은 지정된 차원(여기서는 센트로이드 값이 있는 D0 차원)에서 가장 작은 값의 인덱스를 리턴하는 tf.argmin 으로 결정됩니다. 그리고 tf.argmax 함수도 있습니다.


|함수	|설명|
|--------|---|
|tf.argmin	|지정된 차원을 따라 가장 작은 값의 엘리먼트가 있는 인덱스를 리턴합니다.|
|tf.argmax	|지정된 차원을 따라 가장 큰 값의 엘리먼트가 있는 인덱스를 리턴합니다.|

매 반복마다 알고리즘에서 새롭게 그룹핑을 하면 각 그룹에 해당하는 새로운 센트로이드를 계산해야 합니다. 이전 섹션의 코드에서 아래 코드가 있었습니다.

```python
means = tf.concat(0, [tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])), reduction_indices=[1]) for c in range(k)])
```

* equal 함수를 사용하여 한 클러스터와 매칭되는(역주: 클러스터 번호는 변수 c 에 매핑) assignments 텐서의 요소에 true 표시가 되는 불리언(boolean) 텐서(Dimension(2000))를 만듭니다.
* where 함수를 사용하여 파라메타로 받은 불리언 텐서에서 true로 표시된 위치를 값으로 가지는 텐서(Dimension(1) x Dimension(2000))를 만듭니다.(역주: [Dimension(None), Dimension(1)] 텐서를 만듭니다)
* reshape 함수를 사용하여 c 클러스터에 속한 vectors 텐서의 포인트들의 인덱스로 구성된 텐서(Dimension(2000) x Dimension(1))를 만듭니다.(역주: reshape의 텐서의 크기를 지정하는 파라메타의 두번째 배열요소가 -1이라 앞단계에서 만든 텐서를 차원을 뒤집는 효과를 발휘하여 [Dimension(1), Dimension(None)] 텐서를 만듭니다)
* gather 함수를 사용하여 c 클러스터를 구성하는 포인트의 좌표를 모은 텐서(Dimension(1) x Dimension(2000))를 만듭니다.(역주: [Dimension(1), Dimension(None), Dimension(2)] 텐서를 만듭니다)
* reduce_mean 함수를 사용하여 c 클러스터에 속한 모든 포인트의 평균 값을 가진 텐서(Dimension(1) x Dimension(2))를 만듭니다.



```python
data = {"x": [], "y": [], "cluster": []}

for i in range(len(assignment_values)):
  data["x"].append(conjunto_puntos[i][0])
  data["y"].append(conjunto_puntos[i][1])
  data["cluster"].append(assignment_values[i])

df = pd.DataFrame(data)
sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
plt.show()
```

![pic](https://i0.wp.com/www.jorditorres.org/wp-content/uploads/2016/02/image026.png)

전 내용을 잘 숙지했다면 위 그래프 내용도 쉽게 이해하실 거라 생각합니다.

# Referenece
* http://cs231n.github.io/convolutional-networks/
* https://tensorflowkorea.wordpress.com/5-텐서플로우-다중-레이어-뉴럴-네트워크-first-contact-with-tensorflow/
