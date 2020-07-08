# Bayesian optimization

--------------------------------

	## 개요

​	베이지안 최적화 알고리즘은 모델의 최적 하이퍼파라미터 값들을 찾을 수 있도록 돕는다. 단순히 무작위 추출을 반복하는 것보다는, 기존에 추출되어 평가된 결과를 바탕으로, 추출 범위를 좁혀서 효율적으로 시행하는 것이 더 좋을 것이라고 생각하게 된다. 이러한 아이디어를 Bayesian theory 및 Gaussian process에 접목시켜 개발된 것이 베이지안 최적화 방법이다. 이 방법은 시간대비 성능이 매우 탁월하여, 현재 Amazon SageMaker[5]나 Google Cloud ML과 같은 유명 기계학습 플랫폼에서 주력으로 삼고있는 기술이다. 베이지안 최적화 알고리즘은 아래와 같은 일반적 성질을 갖고 있다. 

* 순차적 접근 방식.
* 목적함수의 도함수를 이용하지 않음.
* 기계학습을 이용하여 목적함수 결과값의 최대값을 예측.
* 목적함수가 노이즈를 갖고 있을 때 사용할 수 있는 최적화 알고리즘.
* 추가적인 데이터의 보강이 이루어질 때, 지속적으로 이용 가능.

## 과정

Bayesian Optimization에는 두 가지 필수 요소가 존재한다. 먼저 **Surrogate Model**은, 현재까지 조사된 입력값-함숫값 점들 (x1,f(x1)),...,(xt,f(xt))(x1,f(x1)),...,(xt,f(xt))를 바탕으로, 미지의 목적 함수의 형태에 대한 확률적인 추정을 수행하는 모델을 지칭한다. 그리고 **Acquisition Function**은, 목적 함수에 대한 현재까지의 확률적 추정 결과를 바탕으로, ‘최적 입력값 x∗x∗를 찾는 데 있어 가장 유용할 만한’ 다음 입력값 후보 xt+1xt+1을 추천해 주는 함수를 지칭한다.

### Surrogate Model

- Gaussian Processes(GP)

  GP는 (어느 특정 변수에 대한 확률 분포를 표현하는) 보통의 확률 모델과는 다르게, 모종의 *함수*들에 대한 확률 분포를 나타내기 위한 확률 모델이며, 그 구성 요소들 간의 결합 분포(joint distribution)가 *가우시안 분포(Gaussian distribution)*를 따른다는 특징이 있다.

### Acquisition Function

- Expected Improvement(EI)

  EI는 현재까지 추정된 목적 함수를 바탕으로, 어느 후보 입력값 xx에 대하여 ‘현재까지 조사된 점들의 함숫값 f(x1),...,f(xt)f(x1),...,f(xt) 중 최대 함숫값 f(x+)=maxif(xi)f(x+)=maxif(xi)보다 더 큰 함숫값을 도출할 확률(Probability of Improvement; 이하 *PI*)’ 및 ‘그 함숫값과 f(x+)f(x+) 간의 *차이값*(magnitude)’을 종합적으로 고려하여, 해당 입력값 xx의 ‘유용성’을 나타내는 숫자를 출력한다.

  ![ei](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/probability-of-improvement-in-gaussian-process-example.png)

  다음 입력값으로 x3x3을 채택했을 시 기존 점들보다 더 큰 함숫값을 얻을 가능성이 높을 것이라는 결론으로 연결한다.

## 결론

![con](http://research.sualab.com/assets/images/bayesian-optimization-overview-1/bayesian-optimization-process.gif)

​	구간 [0.01,0.09][0.01,0.09]에서 최초 3개(n=3n=3), 총 11개(N=11N=11)의 점에 대한 반복 조사 결과니다. 처음 무작위로 점 3개를 두어 Surrogate Model로 함수의 형태를 예측하고,  Acquisition Fuction을 이용해 다음 최적화 값을 구하는 과정이라고 할 수 있다. 