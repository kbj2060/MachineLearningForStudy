1.Multiple features of one input
-----------

![](https://2.bp.blogspot.com/-JEZJhVBlO1w/V2_FOAuVbbI/AAAAAAAAHBw/FUZ8bZQl_dELVkHK6P3Trm7mJxWadWe4wCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

많은 변수를 두어 1차 방정식이 아닌 방정식으로 만든 그래프이다. 변수가 하나이면 상수이고 p+1 개 이면 p개의 서로 다른 변수 x가 필요하다.

![](https://1.bp.blogspot.com/-UUx-lLhmfZU/V2_HkfoHKsI/AAAAAAAAHB8/mj-ALjL6JQsFbyc7Xa84sr7n9c1ddPZnACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

예를 들어 계절에 따라 데이터 값이 규칙적으로 변하는 값에 대해 분석할 때 그래프는 sin과 cos 을 사용하여 그래프를 그를 수 있다.

![](https://2.bp.blogspot.com/-BHD69l9GoGg/V2_I9lhOk0I/AAAAAAAAHCI/Nao8WrahX4Qj5ipknkJv1KWeWtn4cj29wCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

예측 값 yi 는 Feature Extract 을 통해 선택된 변수들의 모임 hi(xi)와 wi에 의해 위와 같이 나타 낼 수 있다. 

2. Incorporating multiple inputs
------------------------------------------

다양한 변수들을 추가해보자.
![](https://3.bp.blogspot.com/-qUktmZVpw7s/V3AAGOkthCI/AAAAAAAAHCY/mDBrvwhfT2EsK58BUn82fBRedSBShXtvgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

예를 들어 화장실 수라는 input value를 추가해 3차원 그래프가 되도록 할 수 있다.
간단한 정의들을 짚고 넘어가야한다.

>Output : y (scalar)
Input : x = (x[1],x[2],...,x[d]) (vector)
x[j] = jth Input (scalar)
hj(x) = jth feature (scalar)
xi = input of ith data point (vector)
>xi[j] = jth Input of data point (scalar)  

![](https://1.bp.blogspot.com/-qAJV36ugksE/V3ACEeAZc2I/AAAAAAAAHCk/knuKvBSSiI8Djzr4PS8jzthXj5FGn4clwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같이 yi를 여러 feature을 통해 구할 수 있다.

![](https://2.bp.blogspot.com/-l-sSPSosY1w/V3YkaLJdDdI/AAAAAAAAHDY/Jy6fNzZPhvkhmPiRvmpAJq6lcLP-mHGLQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

가중치를 구하는 방법이다. 구하고자 하는 가중치를 제외한 나머지 feature을 고정시켜 놓고 구할 수 있다.

3. Setting the stage for computing the least squares fit
---------------------------------------------------
#### STEP 1 Rewrite in matrix notation
![](https://3.bp.blogspot.com/-6TNfBh-kDN8/V3Yl6KdRpZI/AAAAAAAAHDk/Ct76OOhgoaU-KRiG5DpX-4kpxGIM-A9EgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같이 코딩을 할 때 두 벡터를 곱해서 yi에 대한 방정식을 구할 수 있다. 그러므로 yi는 벡터가 된다. yi 벡터는 내가 추정한 값들의 집합이다.

![](https://4.bp.blogspot.com/-ax2inzno-m4/V3Ym6kCZYnI/AAAAAAAAHDw/KYwen8Fdi6INIPvOHM5QxEryPl9oi9k3wCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

이번 슬라이드는 yi에 대한 벡터에 대한 방정식을 만드는 벡터 곱을 표현한 것이다. 즉, yi에 대해 내가 1,2,3...을 넣어보는 것이 아닌 1부터 N까지 모든 y값을 구하는 벡터 곱인 것이다. 

#### STEP 2 Compute the cost
이제 cost function 에 대해 알아볼 것이다.

![](https://2.bp.blogspot.com/-sIOyCxQuBVo/V3YpuRrNcJI/AAAAAAAAHD8/FMxmDFWuOCcZ-Hzs6dODS7XrxcF4pguHgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

RSS 함수가 왜 저렇게 되는 지 알아야한다. yhat 이 h(xi)Tw인 것은 방금 위에서 알아냈다. y-Hw는 나머지 값을 가진 벡터가 된다는 것도 알 수 있다.

![](https://4.bp.blogspot.com/-qgyVwkUk9ww/V3YqN4UWYCI/AAAAAAAAHEA/ZAhKzkrNw54KeFPO2cJCmwZUvHIsv5iowCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

이제 제곱의 형태를 어떻게 표현 할 것인가에 대한 물음에 답해야한다. 그것은 Nx1 형태의 벡터이기 때문에 Transpose 한 벡터(1xN)와 Nx1 벡터를 곱하면 나머지의 제곱 값들의 합을 구할 수 있다. 그것이 곧 RSS(w) 이다.

4. Computing the least squares D-dimensional curve
-------------------------------------------------
#### STEP 3 Take the gradient
![](https://1.bp.blogspot.com/-JsCsJIahgJk/V3aZfc8IzSI/AAAAAAAAHEU/Wn3PYUNqmWULASbBDhnXVSSdQLJyXX0PwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

마지막으로 RSS의 기울기를 구하기 위해 미분하는 과정이다.

#### STEP 4, Approach 1: Set the gradient = 0
![](https://1.bp.blogspot.com/-BJRvqRrt96I/V3aatJxDiKI/AAAAAAAAHEg/-CbFEYVX4X8SYmsYYD4uBoX68IVdidMYgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

여기서 선형대수의 개념들이 나온다.  RSS가 0이 되는 w벡터를 구할 수 있다. 즉, 위와 같은 3D 그래프에서 가장 아랫부분을 구할 수 있는 것이다. 위와 같은 식을 통해 w-hat을  구할 수 있다.

#### STEP 4, Approach 2 : gradient descent
![](https://3.bp.blogspot.com/-MwbNiJjfx7o/V3acjZ5ocMI/AAAAAAAAHEs/28gFRfOs4f0LyNfXjdFFynALIImQfLkeQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

2번째 방법으로는 기울기를 점점 움직여 가면서 최적점을 찾아가는 방법이다. 이 방법을 좀 더 자세히 알아보고자 feature-by-feature update에 대해 알아보자.

![](https://3.bp.blogspot.com/-A8InvCy_niI/V3agWEFkt-I/AAAAAAAAHE4/Dp8iuhWhpIA-ZZHa-tQyOocEhYE9gIXzwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

w를 update 하면서 전체 기울기인 RSS를 조금씩 움직여 최적점을 찾는다.위의 사진을 보면 j번째 feature의 weight를 업데이트하면서 최적점을 찾고 있다.

5. Recap
--------------------
![](https://1.bp.blogspot.com/-RYvba_il78s/V3aiqXdDuRI/AAAAAAAAHFE/S-YZms3VbyI_xw2rHzzQSsBOupcy-BRTACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)
