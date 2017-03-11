1. Scaling ML to huge datasets
-------------

![](https://1.bp.blogspot.com/-2bMN82Gfed0/V63b2zTWtII/AAAAAAAAH3M/AkqVR9Syi94dl-fszdXb4tlxufshpQOCwCK4B/s400/ScreenShot_20160812232309.png)

요즘엔 데이터가 실시간으로 많이 쌓이는 것을 위 그림을 통해 알 수 있다. 나중에 트위터와 iot에 대한 데이터를 나도 하둡을 통해 처리하려 한다. 그에 알맞게 데이터 분석 알고리즘 또한 변화해왔다.

![](https://2.bp.blogspot.com/-SIIto-0BYlU/V63csj_Y5KI/AAAAAAAAH3U/7AMYWeTDa2EVZjr0Wy2gq5EO-E_j3qKAQCK4B/s400/ScreenShot_20160812232309.png)

위와 같이 데이터의 크기에 따라 유행하는 알고리즘의 변화가 뚜렷하다. 요즘 인공지능이 대부분 딥러닝에 의해 성능이 눈에 띄게 좋아졌다고 한다. 또한 데이터의 크기가 커지면서 gradient descent 알고리즘의 한계가 나타나지면서 이제 stochastic gradient 라는 알고리즘이 등장한다.

2. Scaling ML with stochastic gradient
------------

![](https://2.bp.blogspot.com/-X5jpj5GiKb8/V63eXZ9RqMI/AAAAAAAAH3g/nTDsrjEKv7MY_b53KwynR__G06mYFVc4wCK4B/s400/ScreenShot_20160812232309.png)

기울기 하강(gradient descent) 알고리즘을 이용한다면 속도가 너무 느리다는 것을 위 그림이 설명해주고 있다. 유튜브는 하루 50억의 데이터가 생성되는데 기울기 하강 알고리즘을 이용하면 너무 오랜 시간이 소요된다. 그래서 stochastic gradient이 나타났다.

![](https://3.bp.blogspot.com/-8uot-yHCGV4/V63gXToyGlI/AAAAAAAAH3w/jJg-lNfONKAlgdZysXvL4O8M_lAPukdXwCK4B/s400/ScreenShot_20160812232309.png)

stochastic gradient는 기울기 하강 알고리즘과 차원이 다른 알고리즘이 아니다. 간단히 소개하자면, 기울기 하강 알고리즘은 전체 데이터 포인트로 기울기 업데이트를 하지만 stochastic gradient 알고리즘은 하나의 데이터 포인트로 기울기 업데이트를 한다. 아래는 두 알고리즘의 속도를 비교한 것이다.

![](https://2.bp.blogspot.com/-A5eoIcrY2lk/V63hbhjkxcI/AAAAAAAAH34/2ZQA93CYDzITiI-GhIKoDlUUXR6ysZ5NgCK4B/s400/ScreenShot_20160812232309.png)

여기서 stochastic gradient의 문제점은 일관성이 없고 어떤 때는 좋고 어떤 때는 나쁘다는 것이다.

3. Understanding why stochastic gradient works
--------

![](https://2.bp.blogspot.com/-WMIGKXeoeQo/V63jKpz03eI/AAAAAAAAH4I/nyWWbxQFxAAsp68WUcAL_dcjJcIaZipGQCK4B/s400/ScreenShot_20160812235353.png)

위 그래프는 stochastic gradient이 작동하는 방식을 나타낸것이다. 기울기 하강은 한 칸 가는데 전체 데이터를 학습하지만 stochastic gradient는 하나의 데이터로 학습하고 나아간다. 빨간 화살표 중 하나를 고르는 것이다. 어떤 때는 뒤로 가기도 하겠지만 데이터가 노이즈가 적은 데이터라면 수렴하는 방향으로 가는 화살표가 더 많을 것이다. 그러므로 확률적으로 수렴하는 방향으로 가는 화살표가 많다면 마지막에는 수렴할 것이다.

![](https://1.bp.blogspot.com/-OkgbSLSfbSk/V63j7Vs047I/AAAAAAAAH4Q/m6P4dRyrOc4z461flCZC-0JB4f-Hr_8mgCK4B/s400/ScreenShot_20160812235353.png)

수렴하는 path는 위와 같다. gradient 처럼 정확한 방향으로 가는 것은 아니지만 속도가 훨씬 빠르다는 장점이 있다.

![](https://1.bp.blogspot.com/-4LtU6ftwA4k/V63kRfHsG3I/AAAAAAAAH4c/KJmd0LE7iUklFxcSeOO_64AGAC83LDxTACK4B/s400/ScreenShot_20160812235353.png)

4. Stochastic gradient : Practical tricks
---------------
#### 4.1 Shuffle data

![](https://4.bp.blogspot.com/-kxtyH1nsh60/V63mHr4aorI/AAAAAAAAH4o/ZKqOPQzSEbofLDB7UAuzewMSmHH13x36gCK4B/s400/ScreenShot_20160812235353.png)

stochastic gradient은 데이터를 많이 보지 않기 때문에 아주 간단한 규칙이 존재하는 순서라도 모델 정확도에 약영향을 끼친다. 그렇기 때문에 우리는 데이터를 잘 섞어주어야 한다.

#### 4.2 Choosing step size

![](https://1.bp.blogspot.com/-TuwPesVl20M/V63nHjhLfxI/AAAAAAAAH40/90zrRdVESRY5SDrvxzodw3a7ithE-XGawCK4B/s400/ScreenShot_20160812235353.png)

다른 알고리즘과 마찬가지로 너무 큰 step size는 좋지 않다. 적당히 작은 크기가 좋은데 너무 작으면 정확하지만 너무 오래걸린다는 단점이 있다. 이 강의에서는 또 다른 step size 에 대한 팁을 준다. 크기를 반복문을 돌리면서 점점 줄인다는 것이다. stochastic gradient에서 step size를 점점 줄이는 방법이 중요하다고 한다.

![](https://4.bp.blogspot.com/-EG373nmmcVc/V63n0JcEFBI/AAAAAAAAH48/7yTgMw4qO8sI1SAnzgDmiEv4jGkkhqQrQCK4B/s400/ScreenShot_20160812235353.png)

#### 4.3 Don't trust the last coef

![](https://3.bp.blogspot.com/-XTKtata34ps/V63ocs05vyI/AAAAAAAAH5I/ejJvIg_vgbw8E9bVWCMxiH4Bf5nPw1-MgCK4B/s400/ScreenShot_20160812235353.png)

위에서 봤듯이 stochastic gradient의 likelihood가 들쑥날쑥하다. 그러므로 마지막 결과 coef를 믿어서는 안된다. 중간에 더 좋은 likelihood를 가진 coef가 존재할 수 있기 때문이다.

#### 4.4 Batches of data

![](https://3.bp.blogspot.com/-KnJpXqU1baQ/V630_SgGo4I/AAAAAAAAH5Y/oIsc_YLR8vQcxu38rlxwhswk_wtH6KaBwCK4B/s400/ScreenShot_20160812235353.png)

원래는 하나의 데이터로 coef를 업데이트 했으나 이번에는 25개로 늘려 시도해봤더니 2가지 주목할만한 변화가 나타났다.

* 첫째는 25개의 batch로 시도했을 때 더 부드러운 곡선을 가진다.
* 둘째는 수렴되기 전 25개의 batch가 주변을 많이 맴돌지 않고 비교적 바로 수렴된다.

![](https://3.bp.blogspot.com/-PsaU5xlhSEA/V631o82wWYI/AAAAAAAAH5g/ucidw4rWWhohoXtopsTJwaBSVQV8E_DQQCK4B/s400/ScreenShot_20160812235353.png)

위 그래프를 통해 적당한 batch 크기가 중요하다는 것을 알 수 있다.

#### 4.5 Adding regularization

![](https://4.bp.blogspot.com/-HRNVWIh7NFw/V633mIrkl4I/AAAAAAAAH5s/jAxZtl_IJhgI58T9GNyskD8BZq_QORiLACK4B/s400/ScreenShot_20160812235353.png)

선형 회귀에서 배웠던 것처럼 regularization을 여기서도 적용시킨다. L2를 쓸지 L1을 쓸지는 안나와있다.

5. Online learning : Fitting models from streaming data
-----------

![](https://1.bp.blogspot.com/-H6SFqTaB25s/V638Iy9aeQI/AAAAAAAAH54/c_0GP88mq7o5r3iqrPFkF2EKMm1Vzk01ACK4B/s400/ScreenShot_20160812235353.png)

이 때까지 배운 선형 회귀나 분류는 Batch learning이라고한다. 내가 갖고 있는 데이터로 하나의 모델을 만들어 w hat으로 예측한다. 하지만 Online learning 은 다르다. 지정한 시간 단위마다 다른 w hat을 만들어 예측하는 것이다. 이 방법은 시간마다 많은 데이터가 있어야 가능하다. 이 강의에서는 Ad targeting이라는 예시를 든다.

![](https://2.bp.blogspot.com/-Ui8vzXVi8xY/V639laiV6ZI/AAAAAAAAH6E/h4IBN9eawycEiU-tD7CAdMLd3jRjy-ECQCK4B/s400/ScreenShot_20160812235353.png)

ad targeting 의 흐름은 위와 같다. 다른 유저의 광고 데이터를 축적해 내가 클릭한 광고를 통해 나에게 맞는 광고를 제시해준다.