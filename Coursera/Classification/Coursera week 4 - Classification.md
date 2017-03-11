Preventing Overfitting in Decision Trees
=======================

1. Overfitting in decision tree
----------------------------

![](https://4.bp.blogspot.com/-fj_yv82Xh5I/V54wr-HIuvI/AAAAAAAAHr0/w9lxzKeMXpYBFlE_7IzcgotKP5-6n7RswCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

overfitting에 대해 간단히 설명하는 모듈 1이다. training set에 맞추어 계속 learning 을 하다보면 위 그래프처럼 training error는 점점 줄어들지만 true error 와 비슷한 validation error 가 점점 줄다가 다시 증가하는 형상을 보인다.

2. Early stopping to avoid overfitting
--------------------------

![](https://1.bp.blogspot.com/-v0VK9QEY4dU/V54yKJbcssI/AAAAAAAAHsA/V2SGWGtlGPQIKO_TbIyaIu21pAsUG9yiACK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

overfitting 이 되기 전에 트리 만드는 것을 그만두어야한다. 복잡한 트리와 간단한 트리의 에러 차이가 별로 나지 않는다면 간단한 트리를 사용하는 것이 좋다. 그렇다면 어떻게 간단한 트리를 만들 수 있는가?

### 1. Early stopping 
: 복잡한 트리가 만들어지기 전에 그만 만듦

![](https://1.bp.blogspot.com/-b3CdX4yDnK0/V54y7GeceFI/AAAAAAAAHsI/GJWiNvtDin4Z746wYv0-Th13vK-vmLh8ACK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

max_depth까지 트리 복잡도를 늘릴 수 있다. 어떻게 max_depth를 정의할 수 있을까? validation set 이나 cross validation으로 max_depth를 정할 수 있다.

![](https://1.bp.blogspot.com/-HjtCoRKiqFE/V54zpKcY7QI/AAAAAAAAHsY/r23cdCaIY48DQtFiTZEqpAlNs2yjRdGmQCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

트리를 늘려서 classification error를 계산해봤더니 별차이가 없거나 같다면 더 이상 트리를 만들지 않고 위 그림처럼 Risky 값으로 한다.

![](https://3.bp.blogspot.com/-VKyIeVGjR3E/V540tBnKxVI/AAAAAAAAHsk/jJw2rAt3cyogeD7f8h9WaOsE2XOcQeydQCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

데이터 포인트가 너무 작으면 트리를 그만 만드는 것이다. 위 세 방법을 정리하면 validation set으로 max_depth를 정의해 트리 복잡도를 줄이거나 트리를 아래로 더 내렸을 때 classification error가 별 차이 없다면 안 내리는 것이 낫다는 판단 하에 y 값을 정한다.

마지막으로 분류했을 때 노드의 데이터 포인트가 너무 적으면 그 노드의 트리를 더 이상 만들지 않는다.

### 2. Pruning 
: learning algorithm이 종료된 후에 조정하기

![](https://3.bp.blogspot.com/-8EHx_S5y4P0/V59fxpizAPI/AAAAAAAAHtA/TJewdF24qC0-Ybas1vuyk1MXmuYrb-isQCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

logistic regression 과 마찬가지로 total cost 를 구하는데 lambda*L(T)를 통해 조절한다.

![](https://2.bp.blogspot.com/-PNzVYABtnqU/V59gzA5EhzI/AAAAAAAAHtM/VtMUgrBkrPwSsv2EmF5joojYDVLOtEA7ACK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위처럼 람다를 0.3으로 두고 노드를 leaf로 바꿨을 때, total cost 가 감소하므로 prune하는 것이 더 낫다. prune은 가지치기라는 말이다.

Handling Missing Data
=======================================

1. Basic strategies for handling missing data
---------

NA 인 데이터를 어떻게 다룰지에 대해 알아볼 것이다.

### Strategy 1 : Purification by skipping

#### Idea 1 
: Skip data points with missing values

말 그대로 그냥 missing data를 버리는 것이다.

![](https://2.bp.blogspot.com/-zLJRSFnBjzY/V59_QLrDzzI/AAAAAAAAHtY/j3jJVZ2m6oghwVxkZxR4Br67tIb9ECs6ACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

하지만 버린 후의 데이터의 수가 너무 작다면 이 방법을 사용하기 쉽지 않다.

#### Idea 2 
: Skip features with missing values

특정 feature의 missing data가 너무 많다면 그 feature을 지우는 방법도 존재한다.

![](https://1.bp.blogspot.com/-mn_77-BUOdw/V59_2R9KyOI/AAAAAAAAHtg/Xu7apdUT1S0LWCDQDEpRg7lw1I5qLN9hwCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 두 idea는 구현하기 좋다는 장점이 있지만 데이터 손실과 예측 정확도가 떨어진다느 점의 단점이 있다.

### Strategy 2 : Prification by imputing

2번째 방법은 missing data 를 예측해서 맞춰 넣는것이다.

![](https://2.bp.blogspot.com/-BWd6nehduxc/V5-Bs59dt_I/AAAAAAAAHtw/0f1oTPTcSPkU95nZ6H0VBDm8enlq_9dFwCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

확률이 높은 value를 맞춰 넣는 방식이다.


### Strategy 3 : Modify learning algorithm to explicitly handle missing data

![](https://3.bp.blogspot.com/-nNz-wsxgnwM/V5-HGBa5SRI/AAAAAAAAHuA/qRFngy3oi6wWSYPIVjdCn_9Vwf1ciGEFgCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

다음 방법은 트리의 가지에 or unknown이라는 value를 붙여 아래로 내려가도록 하는 것이다. 그렇다면 어떤 가지에 or unknown 을 붙여야하는가?

![](https://3.bp.blogspot.com/-WfGvfyDSe3k/V5-Im70SO9I/AAAAAAAAHuM/k4wXwNKoqd0e2YJFixDxMp5a3e7BGvl6gCK4B/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림처럼 missing data 가 있는 feature에 그냥 다 해보고 error 가 작은 가지에 넣는 수 밖에 없다. 이 방법은 알고리즘을 수정해야하는 단점이 있지만 더 정확한 예측을 할 수 있다.
