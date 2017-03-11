1. Feature selection via explicit model enumeration
-------------------------------------------

이번에는 중요한 feature selection이다. 이 과정은 계산 시간을 줄여주고 정확도를 높여준다. 여기서 짚고 넘어가야하는 것은 Lasso Regression, Ridge Regression은 Linear Regression의 여러 방법 중 하나이다.

Ridge Regression 은 계수(w) 축소를 통해 overfit을 예방하고 정확도를 높이고 Lasso Regression 은 Ridge Regression의 장점을 받아들이고 feature selection을 통해 정확도를 더 높였다. Feature selection은 예측 모델에 영향을 주는 변수들만 따로 구해 모델링하는 것이다. 이제 Feature selection의 방법들에 대해 알아보자.

#### Option 1 : All subsets

이 방법은 변수가 없는 상태로 처음 시작해 점점 변수를 늘려 가며 RSS를 확인한다. 여기서 주목해야할 것은 그냥 N개의 변수 중에 k개를 뽑아서 RSS 확정짓는 것이 아니라 N개 중 k개를 뽑는 모든 경우의 수를 모두 계산해 가장 작은 값을 선택하는 것이다.

그러나! 여기서 가장 작은 값을 뽑는 것이 좋은것인가!?

그렇지 않다.

왜냐하면 모델 복잡도가 증가하기 때문이다.
즉, 트레이닝 셋에 딱 맞는 모델링이 되기 때문에 정작 테스트 셋에는 정확도가 떨어질 확률이 높다.
 
모델 복잡도를 선택하기 위해 여러 방법이 존재한다.

1. validation set으로 평가한다.

2. cross validation 을 수행

3. BIC 같은 모델 복잡도를 줄이는 다른 행렬을 사용

all subset 방법은 직관적이여서 좋아보이지만 너무 많은 계산을 해야한다.

#### Option 2 : Greedy Algorithm

![](https://4.bp.blogspot.com/-BkLaax-vKNs/V4SqwGaJijI/AAAAAAAAHX8/VF70kucOwBAGnJya9kdixnAZkBc2EFEGgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

탐욕 알고리즘의 순서는 위와 같다. 탐욕 알고리즘은 변수를 정하는 순간마다 최고의 퍼포먼스를 보여주는 변수를 선택하는 것이다. all subset과 다른 점은 탐욕 알고리즘은 가장 좋은 퍼포먼스를 보인 첫 변수를 뽑고 2번째 변수를 정할 때 1변수를 정한 상태로 정한다. 하지만 option1 은 그냥 N개의 변수 중 2개를 뽑는 모든 경우의 수를 다 계산하는 것이다.

![](https://1.bp.blogspot.com/-iFRUde1dfVk/V4SsBSpyy3I/AAAAAAAAHYI/4tMf3AMfI0AQ2qSmCm8R9hICvdTuiIoJQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

위의 그림처럼 변수를 늘려가면 에러는 절대 올라가지 않고 option 1 과 결국 같은 에러 값을 갖게 된다. 그렇다면 언제 feature selection을 멈추어야하는가?
 
> 트레이닝 에러가 충분히 작을 때? NO
> 테스트 에러가 충분히 작을 때? NO! 테스트 셋은 건드는 거 아니다.

위에도 얘기 했지만 validation set 이나 cross validation을 사용한다.

![](https://4.bp.blogspot.com/-niJfBIIFYls/V4StnfawUBI/AAAAAAAAHYU/iZggIfwsO8QKJRYLXF7fcL6i90mq1HfgQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

그렇다면 평가 방법의 스텝들에 대해 알아보자. 처음엔 모든 변수의 갯수 D개를 이용해 모델링하고 필요없는 변수를 하나씩 줄여가면서 D-n 모델까지 계산해본다.

얼마나 많은 스텝들을 해야하는가?

- 유저가 정할 수도 있고

- 가장 많게는 D번을 해야한다.

2. Feature selection implicitly via regular regression
------------------------------------------

### Option 3 : Regularize

Regularize 방법으로 저번에 배운 ridge regression이 있다. ridge regression에 대해 좀 더 얘기해야겠다. Thresholding ridge coefficients 에 대해 알아보자.


![](https://2.bp.blogspot.com/-crNQDza9ARs/V4c_UVXq3SI/AAAAAAAAHY8/muPO2x_yyJMPgazRxOej7fHTdkTr0_w-wCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림에서 #bathrooms 와 #showers가 굉장히 비슷한 coefficients를 갖고 있는 것을 볼 수 있다. 상식적으로 둘은 연관이 깊어보인다. 그러므로 아래와 같이 둘을 합친다.

![](https://1.bp.blogspot.com/-2u5Xbp6va10/V4c_b6SVVXI/AAAAAAAAHZA/Z-CzF3LLn00-1UTGKIN3TnqwlbZCCzwwQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)


위 그래프에 대한 이해를 하기 위해 threshold가 한계점이라는 것을 알아야한다. 0 위에 있는 점선이 한계점이고 넘지 못한 변수들을 제거하는 것이다.

그럼 #bathrooms , #sq.ft.living, #sq.ft.lot, #year built , #year renovated, #waterfront 가 변수로 쓰일 것이라는 것을 알 수 있다.

이제 드디어 Lasso Regression을 알아볼 때이다. Ridge Regression과 마찬가지로 람다에 의해 control된다.

![](https://1.bp.blogspot.com/-DlE1DqwqTtc/V4dCJQ8jfXI/AAAAAAAAHZQ/MAEv7WcHGqw_TxtFOo79SM_9mqisxeUcQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

전과 마찬가지로 람다에 의해 cost function이 조절된다.

![](https://2.bp.blogspot.com/-xI4_cer84iE/V4dDBnJJGfI/AAAAAAAAHZc/PyQYjTNlM2Id4VC3NAfj7rEyKDCYvl8VACLcB/s400/%25EC%25BA%25A1%25EC%25B2%25981.PNG)

Lasso와 비교하기 위해 전에 공부했던 Ridge 의 coeffiecient의 변화이다. 그에 반해 아래의 Lasso 의 coeffiecient의 변화를 보면 위의 것과 좀 다르다는 것을 볼 수 있다.

![](https://1.bp.blogspot.com/-x-SSH7SfqOE/V4dDBobH5eI/AAAAAAAAHZg/dRs8QE4l274cKFf3zVHGJCf6LDkz7d_dwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

Lasso는 람다를 조금만 올려도 금방 변수들이 0으로 급락한다. 그리고 중요한 변수의 weight 가 엄청 크다. 즉, 차이가 눈에 띈다.

결론을 말하자면 엄청 중요한 변수인데도 Ridge는 나중에 곧 비슷해지기 때문에 예측할 때 큰 비중을 못차지 하지만 Lasso는 좀만 더 람다를 높여버리면 모두 0으로 수렴해버리기 때문에 중요한만큼 weight를 유지할 수 있다.


3.Geometric intuition for sparsity of Lasso solutions
---------------------------------

Lasso의 cost를 알아보기 위해 우선 Ridge cost를 더 자세히 알아보자.

![](https://2.bp.blogspot.com/-Yps7jJG4A-8/V4dGIug3ESI/AAAAAAAAHZo/-aLkiS4v99sihtLIWgLiTxY_ENxJb3aYACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그래프는 수학적 지식이 있어야 이해가 가능할 거 같다.
우선 x축은 w0 이고, y축은 w1이다.
그리고 보라색 박스가 가리키는 그래프 모형은 타원형이다.
왜냐하면 yi, h0(xi), h1(xi) 들은 상수이고 (보라색 사각형 안의 식) = 상수다. 즉, w0 과 w1 빼고 모두 상수라는 것이다. 그럼 h0(xi)^2*w0^2+ h1(xi)^2*w1^2 = 상수 가 성립한다. 그러므로 위 사각형 안의 식은 타원형이다.

그리고 타원형의 선 안에 있다면 w0과 w1은 달라도 보라색 사각형 안의 값, 즉, RSS 값은 변하지 않는다. 또한 중앙에서 멀어져 갈 수록 RSS가 커지는 것이다. 빨간색 점이 가장 작은 RSS 값을 가진다. w(hat)LS라고도 쓴다.

![](https://1.bp.blogspot.com/-PS3PTKm7rFc/V4dIh8IaAEI/AAAAAAAAHZ4/8989AoTgJMIItaOzvxcLQwu8vldgibtigCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

이제 그 다음 박스를 보면, 이 박스는 좀 더 친숙한 원형 그래프이다. 위에 설명과 비슷한 부분이 많으니 나머지는 생략해도 될 것 같다.

이제 위 두 보라색 박스들을 합친 최종 cost function에 대해 알아보자.

![](https://3.bp.blogspot.com/-uKIS35uPcxs/V4dKjqrcXQI/AAAAAAAAHaE/vrxNOpeNNHY7NxK709-NlK-X-ZlPs-qIwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)


이 그림이 중요한데 그래프가 움직이는 거라 직접 동영상으로 보길 바란다. 저 동영상은 람다가 증가함에 따라 그래프의 움직임을 포착한다. 아까 말한 w(hat)LS가 점점 (0,0)인 점으로 내려오는 것을 볼 수 있다. 즉, coefficient의 변화 그래프에서 본것처럼 점점 0으로 수렴한다.

이번에는 RSS와 magnitude of coefficient의 trade-off 관계에서 람다의 변화에 따른 그래프의 변화를 볼 것이다.

![](https://3.bp.blogspot.com/-GH5KmuXNTjE/V4dNmMvD4GI/AAAAAAAAHaQ/40AgD_rE3GATNvATj8tSGtlesC5KqBsbgCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)


위에서 봐야할 것은 solution 은 여러 가지인데 그에 따른 RSS의 크기와 계수의 크기가 다르다는 것이다.

파란색 펜으로 그린 것은 람다를 크게 하여 계수의 크기를 좀 줄이고 RSS의 크기를 좀 더 늘린 것이다.

여기서 왜 람다를 키우는데 계수의 크기가 주냐는 의문이 들 수 있다.

왜냐하면 람다와 계수의 크기는 서로 반비례 관계이기 때문이다.

즉, a*x = 4에서 a 가 커지면 x 는 줄어들 수 밖에 없다.

드디어 이제 Lasso에 대해 똑같은 방식으로 알아보자.

![](https://4.bp.blogspot.com/--ewPiaDb_RU/V4dPkADdHjI/AAAAAAAAHac/Xw7QjbRu-ec42CEdazQRzn7NIkpRzcKDACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)


Lasso의 RSS는 Ridge 와 같기 때문에 생략하고 저 위에 있는 보락색 상자의 식을 보자. 저 식은 다이아몬드를 그리는 그래프를 나타낸다. 이것도 고등수학에 나온다. 가운데 빨간색으로 그려진 값이 가장 작은 값을 나타낸다. 다시 람다를 증가시킴에 따라 cost function의 변화를 볼 것이다.

![](https://3.bp.blogspot.com/-UC92mwX8zW4/V4dRRankJ5I/AAAAAAAAHao/LmETmUNFYjU0UkER4xEZV3Db3y8TjIz_QCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

우선 눈에 띄는 Ridge Regression과의 차이점은 y축인 w1의 값이 짧은 순간에 0 값이 된다.

점점 같은 속도로 (0,0)으로 수렴되는 것이 아니라 Lasso는 거의 시작과 동시에 w1값이 0으로 된다. 그리고 w(hat)값이 (0,0) 이 된다. Ridge Regression은 거의 (0,0)이였다는 점과 다르다.

이번에도 RSS와 magnitude of coefficient의 trade-off 관계에서 람다의 변화에
따른 그래프의 변화를 볼 것이다.

![](https://2.bp.blogspot.com/-BKx4ImHoJCY/V4dSb-UD-CI/AAAAAAAAHaw/uRDuWp02zQcu4R1b4sEpKHeCNvYiM02kwCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

Ridge와 다이아몬드 그래프를 제외하면 설명이 동일하다. 하지만 위와 같은 모든 변화는 2차원에서 보였지만 차원이 증가할수록 복잡해진다.

4.Setting the stage for solving the Lasso
------------------------------------

#### Aside 1 : Coordinate descent

![](https://1.bp.blogspot.com/-u2UnyfwnFKE/V4eGxTsA_mI/AAAAAAAAHbE/HwkxGz9Bo0EiVLIS_2apfX203Lxbhw7EQCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림처럼 Coordinate descent 는 직역하면 좌표 하강이다. 즉, 위와 같은 현재 w0과 w1의 타원형 그래프에서 w0을 고정시키고 w1을 최소화하고 w1을 고정시키고 w0을 최소화시키는 과정을 반복하다보면 최소값에 수렴하게 되는 방법이다.

![](https://2.bp.blogspot.com/-VqblX9x9myQ/V4eIVqaVEeI/AAAAAAAAHbM/TiC8e4HNd-wwI9-inB9Thb18_9lFL94CwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

이 Coordinate descent가 Gradient descent와 뚜렷한 차이는 step size가 없다는 것이다.

Gradient descent는 step size에 민감에 조금만 커도 Global minimum을 찾지 못하고 넘어가는 경우가 발생했기 때문에 step size가 없다는 것은 간편하다. 또한 위와 같은 많은 문제를 헤쳐나가기 좋은 접근 방법이다.

#### Aside 2 : Normalizing features

![](https://4.bp.blogspot.com/-wSp2gt00eUU/V4eKCUvueDI/AAAAAAAAHbU/7Z2s1elbCAc3SfAiIM6jg3pf5wAAvU6FACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

행이 아닌 열을 정규화하는 과정이다. 열을 한 feature의 값들의 집합이다.
여기서 중요한 것은 트레이닝 셋과 함께 똑같은 정규화 공식으로 테스트 셋에도 적용시켜야 한다는 것이다.

#### Aside 3 : Coordinate descent for unregularized regression (for normalized feature)

![](https://4.bp.blogspot.com/-bth3jTTqncE/V4euwPsJ_5I/AAAAAAAAHbk/k0oyQcek-isAXFPw6K5hQC-hvw0lVk29gCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같이 normalized feature을 이용해서 RSS의 미분값을 계산해봤다.
그리고 마지막의 -2pj+ 2wj = 0 으로 두고 LS를 수행한다.
LS에 대해 자세히 설명해놓은 블로그가 있어 주소를 적어둔다.

http://darkpgmr.tistory.com/56

5.Optimizing the Lasso objective
--------------------------------------

라소를 최적화하는 과정에 대해 알아볼 것이다.

![](https://4.bp.blogspot.com/-VVp5mdblFhU/V4hVlSTdFTI/AAAAAAAAHcA/QLHNL5BTJL0o3tzzpmhtri2J-8VXdJf8QCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같이 Lasso의 coordinate descent를 정의했다.
이제 ridge regression과의 차이점에 대해 알아보자.

![](https://3.bp.blogspot.com/-oRT1emXN_FI/V4hWgxtmjpI/AAAAAAAAHcM/AgrQ-bXoyXs60pIxeujCZPXvCRZ2NMEaACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같은 그래프는 Soft thresholding 을 이용해 만든 그래프이다.
헷갈릴 수 있으니 x축과 y축 먼저 보고 그래프를 보는 것이 좋다.
위 그래프를 보면 점선이 w hat ridge 이고
연두색은 -2pj + 2wj = 0 의 LS를 나타낸 것이고
보라색은 w hat lasso인 것을 알 수 있다.

![](https://3.bp.blogspot.com/-VX-g43u43zY/V4hYHN3Lk6I/AAAAAAAAHcY/9XRjFHROsDE6m5d9FMu6-6awDO-kRJocQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림처럼 coordinate descent는 lasso를 풀 수 있는 여러 가지 방법들 중 한 가지이다.

![](https://1.bp.blogspot.com/--NYGXsPzdkc/V4hZLRVEyBI/AAAAAAAAHck/eeDDUJVHHA4OcQ0R1TcCjUaiqaqnq4a8ACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림은 정규화되지 않은 변수들을 coordinate descent 을 수행한 것이다. w hat j를 구할 때는 normalizer인 z j 를 통해 식을 표현한다. 위의 방법이 훨씬 깔끔한 것을 알 수 있다.


6. Tying up loose ends
----------------------------------

이번에는 어떻게 람다를 구할 것인가에 대해 공부한다. Ridge Regression에서와 같이 데이터가 많으면 validation set을 이용하고,
데이터가 별로 없으면 cross validation을 이용할 수 있다.
Lasso에서 할 수 있는 방법에 대해 공부하고자 한다. 우선 Debiasing Lasso에 대해 알아보자.

![](https://3.bp.blogspot.com/-rJ4gadlHrNw/V4hTbEDkJ6I/AAAAAAAAHb0/OKhhGL5AAZUQu5LP9VcjhODybrPmXxxJwCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림을 이해하면 Debiasing Lasso의 위력을 볼 수 있다. 간단하게 Debiasing Lasso는 말 그대로 bias를 줄이는 절차를 통해 에러를 줄이는 방법이다.
첫번째 열은 True coefficients를 이용해 0이 아닌 변수 4096 변수 중에 160개만 사용한 것것이고
두번째 열은 L1reconstruction을 통해 1024개의 변수를 사용한 예인데 에러율이 0.0072이다.
다음은 Debiasing Lasso를 통해 1024개의 변수를 사용했더니 에러율이 3.26e-005로 현저히 떨어진 것을 볼 수 있다

![](https://1.bp.blogspot.com/-J0PgGWs0WZM/V4hgASc1ezI/AAAAAAAAHc0/oSwNZAHFeUc5_m2l_xhhbD3X20y021EFgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

Lasso regression에 대한 이슈들에 대해 알아보자.
우선 라소는 아주 관련이 깊은 변수들의 그룹이 있을 때 그룹 중 마음대로 변수를 고른다.
두번째로 경험적으로 ridge가 예측하는 데에 더 좋은 퍼포먼스를 보인다.
그래서 Elastic net이라는 하이브리드 regression이 사용된다. 즉, ridge와 lasso를 섞은 느낌이다.
