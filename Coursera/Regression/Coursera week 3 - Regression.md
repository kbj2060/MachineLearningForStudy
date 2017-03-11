1.Defining how we assess performance
-------------------------------------------
![](https://3.bp.blogspot.com/-_sZPN0JHwck/V3-IQ1mEzkI/AAAAAAAAHIA/FEDX6qvY9PsXiBH3Wcj65MjmyQ6SGhZKQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그림처럼 model&nbsp;+ algorithm 으로 fitted function 으로 만들고
Predictions 를 통해 결정을 하고 수익을 얻는 구조이다.

![](https://3.bp.blogspot.com/-St4w8j5dKNs/V3-I6s7Wl9I/AAAAAAAAHIE/NtKgm3WzEUwXTkshoYehap-hja0h23_fgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

손실 함수를 위와 같이 표기한다. 위 그림에서 보듯이 fhat함수는 예측값이고 y값은 실제 값이다. 저번 게시물에 적었듯이 loss 함수를 만드는 것이다. 밑의 예처럼 두가지 방법이 있다. 절대값 에러와 제곱 에러이다.

2. 3 measures of loss and their trends with model complexity.
--------------------------------------

### part 1 : Training error
![](https://3.bp.blogspot.com/-opCaqcqwzRs/V3-MqH0M8vI/AAAAAAAAHIU/eQwLGlEYC8UyswAxptLFpsj2GE7mudalACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

파란 점은 트레이닝 셋에 포함된 점들이고 나머지는 포함되지 않은 점들이다. 트레이닝 셋에 따라 그래프를 그린 사진이다. 그리고 계산한 에러들을 Training Error라고 한다. 트레이닝 에러들을 보통 위에 있는 제곱을 이용해 계산한다.

![](https://2.bp.blogspot.com/-d_qVTGyD9GI/V3-N6QDikzI/AAAAAAAAHIg/JdOZO8H0NC4u8d_fCcraHQFS9MU8i3YeACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

위의 사진처럼 트레이닝 에러와 모델 복잡도는 서로 반비례 관계에 있다.
하지만 작은 트레이닝 에러 값이 좋은 예측을 한다는 것은 아니다. 왜냐하면 트레이닝 셋의 데이터들이 모든 데이터를 대표하는 것이 아니기 떄문이다.


### part 2 : Generation (true) error
![](https://4.bp.blogspot.com/-PkJZ_5DebMI/V3-0ze96O0I/AAAAAAAAHIw/RnPBPxGQwToCN7bwzMNNtAV6QILFTdITwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

part 1 에서는 training error 와 모델 복잡도에 대해 알아봤고 이번에는 generation error에 대해 알아보면 위와 같은 그래프를 하고 있다. 또한 training error는 계산할 수 있으나 generation error 는 계산할 수 없다.

### part 3 : Test error
![](https://1.bp.blogspot.com/-Mfm6R9BtCHs/V3-2IxhqnpI/AAAAAAAAHI8/YY-EtbHKJgEM75FNJAmB8SW_Mxa0cAnSACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

거의 정확하게 test error를 통해 generation error 추측할 수 있다.&nbsp;
Test error 는 아래의 식과 같이 테스트 셋에서 모든 데이터에서의 손실의 평균이다.

![](https://2.bp.blogspot.com/-gpTekKCipY0/V3-3BDaSfYI/AAAAAAAAHJI/cCG7B48AdMU12ZU_IoL0T0D_yuwS-Aw5gCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

위에서 얘기한 것들을 모두 합치면 위와 같은 그래프라고 할 수 있다. test error 그래프가 true error 와 거의 비슷하다는 것을 볼 수 있다.

![](https://4.bp.blogspot.com/-geIQ8z1R1jE/V3-4E1TR-8I/AAAAAAAAHJU/cGlvrHh7Sa4ItvTP5buwZb4OVvJbiVR5QCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

Overfitting 를 에러 그래프를 통해 알아보면 w^이라는 예측값과 그보다 작은 모델 복잡도를 가진 w`이 있을 때,

* traning error 관점에서 w^  <  w` 

* true error 관점에서 w^  >  w`

이라는 두 조건을 만족시키면 w^은 w`보다 더 overfitting 되어 있다고 한다.

![](https://3.bp.blogspot.com/-uhO3fJa5tNc/V3-5ncVaLlI/AAAAAAAAHJg/YMV82w-28XYUY8IGt4sZN2HKhD8VejhtACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 사진처럼 트레이닝 셋과 테스트 셋을 분리시켜 트레이닝 시켜야한다.
트레이닝 셋이 더 많고 테스트 셋이 그보다 더 적은게 더 효율적이다.
이 방법은 많은 데이터가 있을 때 좋다. 하지만 만약 적은 데이터를 갖고 있을 경우 cross validation이라는 방법을 쓰는 것이 좋다.




3. 3 sources of error and the bias-variance tradeoff
------------------------------------------------
![](https://3.bp.blogspot.com/-EiUXlr1N1Nc/V4DaCjFQVmI/AAAAAAAAHJw/0uXl-0tuB-EgmpB60CVeYWjOKUAX1jFEwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

예측하는데 3가지 종류의 에러가 존재한다.
하나씩 알아보자.

#### Part 1 Noise

![](https://4.bp.blogspot.com/-CTvFHl4KHyQ/V4DapQ8OnVI/AAAAAAAAHJ0/N1sHL_TS1lMY7OJCznO1jUAl6LkoYDG8QCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

첫번째는 noise이다. 노이즈는 함수에서 앱실론으로 쓰이며, 현실과 이론의 차이라고 할 수 있을 것 같다.
즉, 더 좋은 모델을 쓴다고 노이즈를 줄일 수도 없다.
이건 노력 밖의 에러니 그냥 이런게 있다고만 알고 가자.
그리고 나머지 두 에러에 집중하자.

#### Part 2 Bias

![](https://1.bp.blogspot.com/-E9uczavl1oI/V4D81rFTSKI/AAAAAAAAHKI/In5Sw871h50tjpelnlIDcUuQ2zzsVr5XQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)]

f slash(w) 는 여러 다른 트레이닝 셋을 통해 fit 을 한 함수들의 평균 값을 나타낸 선이다. 그리고 최종적으로 Bias 는 True 함수와 slash(w) 함수의 차이이다.
Bias 는 '학습 모형이 입력 데이터에 얼마나 의존하는가'를 나타낸다고 할 수 있다.
Bias, 즉 선입관이 크면, (좋게 말해서) 줏대가 있고 (나쁘게 말해서) 고집이 세기 때문에 새로운 경험을 해도 거기에 크게 휘둘리지 않는다. 평소 믿음과 다른 결과가 관찰되더라도 한두 번 갖고는 콧방귀도 안 뀌며 생각의 일관성을 중시한다. 
반대로 선입관이 작으면, (좋게 말하면) 사고가 유연하고 (나쁘게 말하면) 귀가 얇기 때문에 개별 경험이나 관찰 결과에 크게 의존한다. 새로운 사실이 발견되면 최대한 그걸 받아들이려고 하는 것이다. 그래서 어떤 경험을 했느냐에 따라서 최종 형태가 왔다갔다한다. (High Variance, Low Bias)

#### Part 3 Variance

![](https://3.bp.blogspot.com/-OYvw-M4O5PA/V4D_dJb4EtI/AAAAAAAAHKU/62wsvvIUqUg3FxDOKNZz0FsW2YeRZNooACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같이 여러 트레이닝 셋을 통한 polynomial fit 들의 평균은 f slash(w) 처럼 비교적 평평한 그래프를 나타내고 그래프에서 보듯이 variance 의 크기가 엄청 크다는 것을 알 수 있다. 그러므로 높은 복잡도는 결국 높은 variance 를 나타 낼 수 밖에 없다.

#### part 4 Bias-variance trade-off
![](https://4.bp.blogspot.com/-wSjUPqewADk/V4EBOnjhhtI/AAAAAAAAHKg/KmHeNLVgGOEYoUjw5he7PLVPNP4iIrtoQCLcB/s320/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위와 같이 MSE를 만들 수 있다. MSE의 가운데 최소값에 다가가는 것이 목표이다. 하지만 우리는 모델을 만들 때 MSE를 구할 수는 없다. 왜냐하면 저 공식의 기반은 True error 여서 구할 수가 없다.  True error를 구할 수 없는 이유는 전에 배웠다. 그러나 나머지 코스를 배우면서 여러 방법으로 bias와 variance 의 트레이드 오프 관계를 최적화하는 방법을 배울 것이다.

![](https://3.bp.blogspot.com/-FZcDyWyScOY/V4EC2uavnqI/AAAAAAAAHKs/5l5ox6zIbkkhXfZoTz3NdK1_6M-pZrTPwCLcB/s320/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 그래프를 보면 에러와 데이터 수의 상관 관계를 보여준다. 우선 true error 는 데이터 수가 많아질수록 에러는 줄어드는 것을 보여준다. 하지만 bias + noise 값에 수렴할 수 밖에 없다. 아래의 training error 가 위 그래프처럼 아래 부분에서 시작한다면, 데이터 수가 많아질수록 어느 값에 수렴하는 것을 볼 수 있다.



4. Putting the pieces together
--------------------------------------------------
![](https://3.bp.blogspot.com/-5zOaz3KtvQ0/V4EEIs559UI/AAAAAAAAHK4/nI5yjr3-c-0szbWLrkU5diIIZZ35RG7xACLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

이번 3 주에서는 모델을 선택할 때 모델 복잡도를 신경써서 파라미터들을 조절하는 것을 배웠고 그리고 모델 에러 평가에 대해 배웠다. 여기서 중요한 것은 generation error 는 구할 수 없으나 test error 를 통해 추정할 수 있다는 것이다.

![](https://1.bp.blogspot.com/-8Oyq0ciLLj8/V4EFu_Fq8PI/AAAAAAAAHLE/KO5OovCpk8gVGj6RqdjR1ug5F1CL8Uu4gCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

또한 위처럼 트레이닝 셋을 통해 w^를 구하고 구한 w^을 밸리데이션 셋을 통해 람다를 구한다. 람다는 방금 전에 본 복잡도에 영향을 주는 파라미터다. 마지막으로 테스트 셋을 통해 제너레이션 에러를 평가한다.