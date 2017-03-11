1. Regression fundamentals
------------------------------------------------
[![](https://3.bp.blogspot.com/-YP1Yi1Sf2xY/V0CC6k3KfXI/AAAAAAAAGrk/hj-3IQqXgNQ25zVdxerOuTSan8PmNtqSACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/-YP1Yi1Sf2xY/V0CC6k3KfXI/AAAAAAAAGrk/hj-3IQqXgNQ25zVdxerOuTSan8PmNtqSACLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

집의 크기를 x값으로 두고 가격을 y값으로 둔 좌표 평면으로 집의 크기로 집의 가격을 예측해보는 예제이다. 그래프에 수식에 따른 정의를 잘 알아두어야겠다. xi , yi , f(xi) , f(x) 에 대한 차이점을 잘 숙지해두자.

[![](https://1.bp.blogspot.com/-E7aEVW7l-EQ/V0CEySkyQpI/AAAAAAAAGr0/haQd4_ZDvMgr0f6FD-kT3TjBM03w1iGeQCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-E7aEVW7l-EQ/V0CEySkyQpI/AAAAAAAAGr0/haQd4_ZDvMgr0f6FD-kT3TjBM03w1iGeQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

Regression의 flow chart 이다. 여기서 x,yhat,fhat,y에 대한 정의를 잘 숙지해두자. 알아도 다시 한 번 차근차근 다시 보자.

2. The simple linear regression model, its use, and interpretation
--------------------------------------------------------

[![](https://3.bp.blogspot.com/-HBh2bQHfvHU/V0CFPOg8BiI/AAAAAAAAGr8/Wbm1JakUTyAkfTl2nNeiwNQP3Zl44wkigCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/-HBh2bQHfvHU/V0CFPOg8BiI/AAAAAAAAGr8/Wbm1JakUTyAkfTl2nNeiwNQP3Zl44wkigCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

첫번째 그림에서 봤던 일차 방정식 yi를 더 자세히 나타낸 것이다. 앱실론에 대한 내용을 잘 기억하자.
[![](https://1.bp.blogspot.com/-6lInY5GdFdk/V0CGTjA3h6I/AAAAAAAAGsM/FHoDBrlp9t8E-yDPHJ_RG8QBtI3twfVQwCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-6lInY5GdFdk/V0CGTjA3h6I/AAAAAAAAGsM/FHoDBrlp9t8E-yDPHJ_RG8QBtI3twfVQwCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

공부하다가 RSS 를 계속 까먹어서 고생을 많이 했다. 에러 값의 제곱을 모두 더한 것이다. 그럼 에러 값이 모두 양수 값이 되서 Regression 모델들에서 RSS값이 작은 것이 가장 좋은 Regression model이 되는 것이다.

[![](https://1.bp.blogspot.com/-hytqvkcK3kc/V0CHd4viSRI/AAAAAAAAGsY/yFrzzZwrl50GEpDx997WRPAUupvUqdn8gCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-hytqvkcK3kc/V0CHd4viSRI/AAAAAAAAGsY/yFrzzZwrl50GEpDx997WRPAUupvUqdn8gCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림이 전체적인 과정이다. 내가 원하는 평수의 집은 얼마일까? 라는 질문에 답해줄 수 있는 간단한 Regression model 이다. w0와 w1의 parameter 들의 값을 정하는 부분은 다음에 나올 것이다.

3. An aside on optimization: one dimensional objectives
----------------------------------------------------------

[![](https://4.bp.blogspot.com/-2zUvE-S_ZL4/V0CKTfmTcQI/AAAAAAAAGsk/yrMTqFpYfW0zQMqUO4YT7rJeiir_sL1xACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://4.bp.blogspot.com/-2zUvE-S_ZL4/V0CKTfmTcQI/AAAAAAAAGsk/yrMTqFpYfW0zQMqUO4YT7rJeiir_sL1xACLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

이번에는 위 그림에서 how? 부분인 w0,w1 값을 어떻게 정할 것인가? 에 대해 공부 할 것이다. 
 
 [![](https://4.bp.blogspot.com/-wZ_21B932ps/V0CK43j0pqI/AAAAAAAAGso/HKdykqw4l6wbX4F2K4RfbDOM2xqpd0rtACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://4.bp.blogspot.com/-wZ_21B932ps/V0CK43j0pqI/AAAAAAAAGso/HKdykqw4l6wbX4F2K4RfbDOM2xqpd0rtACLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
 
Optimization 에서 RSS의 최소점인 빨간점의 w0과 w1 값이 최적화 된 parameters 이다. 여기서 잠깐 Concave 와 Convex function에 대해 알고 가자. Concave 는 오목한 그래프이고( ex) -x^2 ), Convex 는 볼록한 그래프이다.( ex) x^2 )

 
 [![](https://1.bp.blogspot.com/-95T5WZhvjNM/V0CMZ05qXNI/AAAAAAAAGs4/XwNUfpFeNdQZ5d4hFHbmkk1HuTcHnhhlgCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-95T5WZhvjNM/V0CMZ05qXNI/AAAAAAAAGs4/XwNUfpFeNdQZ5d4hFHbmkk1HuTcHnhhlgCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
  
 **이것을 왜 알아야할까?** 
 
전 Optimization부분에서 RSS가 최소가 되는 값을 찾을 때처럼 극솟값이나 극댓값을 찾아야 할 떄 Concave 인가 Convex 인가에 따라 과정이 다르기 떄문이다. 이런 것들을 통해 local minimum , global minimum 과 같은 용어를 이해할 수 있다. 극솟값과 극댓값 구하는 것들은 고등 수학 과정에 나오므로 생략한다. 
(이럴 땐 대한민국 교육과정이 좋다..) 
 
[![](https://1.bp.blogspot.com/-NulzFamLHm4/V0COcS1Vc3I/AAAAAAAAGtE/vwXeDWls_0UX9mvk6G7EMJXQScWTMzYhACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-NulzFamLHm4/V0COcS1Vc3I/AAAAAAAAGtE/vwXeDWls_0UX9mvk6G7EMJXQScWTMzYhACLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
 
극솟값을 구하는 알고리즘이다. step size 를 줄이면서 순간 기울기가 0이 되면 반복을 그만하고 극솟값을 찾은 것이다. 
 
 [![](https://1.bp.blogspot.com/-Pc3cr82iyWE/V0CP9Mf9DOI/AAAAAAAAGtQ/4mZoeEZBWYg0oQo5TA16vjtdDejGhy9-gCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-Pc3cr82iyWE/V0CP9Mf9DOI/AAAAAAAAGtQ/4mZoeEZBWYg0oQo5TA16vjtdDejGhy9-gCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
 
자, 이것을 위해 간단한 오목한,볼록한 그래프부터 시작한 것이다. w0,w1,...,wp의 기울기를 벡터화시킨 것이다. 아래 예제를 보는 것이 더 이해가 잘 될 것이다. 
 
[![](https://1.bp.blogspot.com/-fJCMX4W30PE/V0CQ7C5uGuI/AAAAAAAAGtY/LafSwshIKBcKs65YjoKpv8bqtC6DPQYPgCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-fJCMX4W30PE/V0CQ7C5uGuI/AAAAAAAAGtY/LafSwshIKBcKs65YjoKpv8bqtC6DPQYPgCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
 
고등 수학에서본 x,y가 있는 함수에서 x에 대한 미분을 할 때는 y 를 상수 취급해 계산하는 것이다. 위와 같은 계산을 통해 기울기 벡터를 만들 수 있다. 
 
[![](https://3.bp.blogspot.com/-Y6pR8gbXeMY/V0CSmjH-qWI/AAAAAAAAGtk/lCUttKcJg_oxN815qNFJpNhlP4LB1vuaACLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/-Y6pR8gbXeMY/V0CSmjH-qWI/AAAAAAAAGtk/lCUttKcJg_oxN815qNFJpNhlP4LB1vuaACLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
  
3차 그래프에서 2차 그래프로 옮겨온 것이다. 즉 , 3차원 그래프를 z축과 평행하게 슬라이스를 내어 잘린 단면을 2차 좌표평면으로 옮겨온 것이다. 위와 같은 알고리즘은 원래 보라색이 기울기 방향인데 - 값을 두어 (벡터의 - 는 방향이 반대 ) 점점 수렴하도록 하는 알고리즘이다.

3. Finding the least squares line
--------------------------------------

[![](https://3.bp.blogspot.com/--XaJZLlvo48/V0E9OErFtSI/AAAAAAAAGt8/rw3YRwnzdYMV2VlH2Hqi6m2MFIOQ0Yb7wCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/--XaJZLlvo48/V0E9OErFtSI/AAAAAAAAGt8/rw3YRwnzdYMV2VlH2Hqi6m2MFIOQ0Yb7wCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
여기서 gi(w) 는 RSS 의 함수이다. RSS함수를 미분해 극솟값을 찾아가기 위한 계산을 할 것이다.

[![](https://3.bp.blogspot.com/-IF0-q90_ZcA/V0E-WgHkLGI/AAAAAAAAGuI/H2rM4L0b9HAdD777LpS65E9QfQP3IZq_wCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/-IF0-q90_ZcA/V0E-WgHkLGI/AAAAAAAAGuI/H2rM4L0b9HAdD777LpS65E9QfQP3IZq_wCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

여기서 RSS(w0,w1) 함수는 w0 에 대해 미분한 첫 번째 값과 w1 에 대해 미분한 두번째 값이 존재한다. 여러 approach 로 극솟값을 찾아볼 것이다.

[![](https://2.bp.blogspot.com/-uCkNRN9YybA/V0FEc1jgHJI/AAAAAAAAGuY/xjrKEat-DM42Eq9-BS78aEgkucsYeUDpgCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://2.bp.blogspot.com/-uCkNRN9YybA/V0FEc1jgHJI/AAAAAAAAGuY/xjrKEat-DM42Eq9-BS78aEgkucsYeUDpgCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

첫 approach 는 gradient 를 0으로 설정하는 것이다.  RSS(w0,w1)을 모두 0으로 설정하면 그림의 top term 처럼 w0hat 값을 얻을 수 있다. w0hat = (집의 가격의 평균) - (기울기 추정값 * 집 평수의 평균) 
w1hat 은 w0hat 을 이용해 위 그림의 bottom term처럼 구할 수 있다. 
(여기서 hat은 추정값이라는 뜻이다.) 
하지만 이해가 잘 가지 않았다. 그래서 다시 정리하자면 우리가 흔히 사용하는 y를 x에 관해 나타낸다고 해보자. 


예를 들어 y = x^3-4*x^2+5 라는 함수가 있다. 
그렇다면 y의 극솟값을 찾기 위해 우리는 x에 관해 미분해 
y' = 3*x^2-8*x = 0 에서 x = +3/8 or -3/8 값을 찾아낸다. 
위 그림도 똑같다. 단지 다른점은 y 가 RSS 로, x 는 w0 or w1 로 바뀌었을 뿐이다. RSS 에 대해 w0 로 미분하고 w1 으로 미분해 극솟값을 찾는 과정이다.

[![](https://1.bp.blogspot.com/-R8ILMBIAkHI/V0Gs0TtKxWI/AAAAAAAAGvY/fVp0vEFS3NodBaR75l_HwmKTW8XLAAwGwCLcB/s640/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-R8ILMBIAkHI/V0Gs0TtKxWI/AAAAAAAAGvY/fVp0vEFS3NodBaR75l_HwmKTW8XLAAwGwCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 


두번째 approach 는 Gradient descent 이다. 왼쪽에 있는 좌표 평면의 가장 안쪽에 있는 원으로 수렴할 때까지 RSS의 미분값을 뺸다. 여기서 step size 와 convergence criteria 를 꼭 설정해서 수렴하도록 반복문을 돌려야한다. 