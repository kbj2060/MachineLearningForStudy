 1. Course overview and details
----------------------------------------------

[![](https://1.bp.blogspot.com/-fbhSDCBbeXs/V5cDYvqHfLI/AAAAAAAAHjc/JGrhwWBXWlgyPA1Y7zwjUmivhoDAguuPwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-fbhSDCBbeXs/V5cDYvqHfLI/AAAAAAAAHjc/JGrhwWBXWlgyPA1Y7zwjUmivhoDAguuPwCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

위 과정은 분류 과정의 overview이다. 여기서 내가 주목한 것은 linear classifiers 와 logistic regression의 차이점이다. 그 차이점을 단적으로 보여주는 아래의 두 그래프를 보자. 
 
[![](https://3.bp.blogspot.com/-SgrmJlGL8Mk/V5cEHiSqkgI/AAAAAAAAHjg/dFPNRR7KOtcEW5cYt1DHPp8JEgAy3xrvQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/-SgrmJlGL8Mk/V5cEHiSqkgI/AAAAAAAAHjg/dFPNRR7KOtcEW5cYt1DHPp8JEgAy3xrvQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
<div class="separator" style="clear: both; text-align: center;">

[![](https://3.bp.blogspot.com/-tLGylfdMWIw/V5cENhvh1qI/AAAAAAAAHjo/RP9nDn7ICUMZSY0qO2mRuGs8sqryv9hLQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://3.bp.blogspot.com/-tLGylfdMWIw/V5cENhvh1qI/AAAAAAAAHjo/RP9nDn7ICUMZSY0qO2mRuGs8sqryv9hLQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
[](https://3.bp.blogspot.com/-SgrmJlGL8Mk/V5cEHiSqkgI/AAAAAAAAHjg/dFPNRR7KOtcEW5cYt1DHPp8JEgAy3xrvQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG)

첫번째 사진이 linear classifier이고 두번째가 logistic regression 이다. 자세히 보지 않고 단적인 차이점은 선의 유무이다. 전자는 선을 경계로 위는 - 아래는 +로 표시하고 후자는 색의 진함으로 + 혹은 - 일 확률을 표시했다.  후자의 1 / ( 1+ e^(-w*h(x)) 는 시그모이드함수이다. 다른 여러 모델들과 알고리즘들을 머신러닝 인 액션이라는 책에서 한번  훑어본 적이 있어서 그런지 굉장히 익숙하다. 이 강의를 통해 다시 한 번 개념을 정리하고 실습해봐야겠다. 

 

2. Linear Classifiers
------------------------------------------
선형 회귀에서는 부동산 가격에 대한 예제를 들었지만 이번 분류에서는 스시 가게의 리뷰를 보고 단어들을 분석해 리뷰의 내용이 좋은 것인지 나쁜 것인지 분류하는 예제에 대해 공부한다. 

[![](https://1.bp.blogspot.com/-8u8hPfr707U/V5cHKIPXQeI/AAAAAAAAHj0/yYZClEekS7Quf_MQQq1sH_sHITBoVq8hwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-8u8hPfr707U/V5cHKIPXQeI/AAAAAAAAHj0/yYZClEekS7Quf_MQQq1sH_sHITBoVq8hwCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 
 

분류의 기본 알고리즘은 위와 같다. 단어들에 weight를 두어 리뷰의 단어들을 계산하여 score를 책정한 뒤 0보다 크면 좋은 리뷰라는 +1을 y hat 에 저장하고 0보다 작으면 나쁜 리뷰라는 -1을 y hat 에 저장한다. 

[![](https://2.bp.blogspot.com/-8rcRSea9rOk/V5cI6ZI4LBI/AAAAAAAAHkA/647G8vnsTw0bE5KHbmdyaVen1veDodAiQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://2.bp.blogspot.com/-8rcRSea9rOk/V5cI6ZI4LBI/AAAAAAAAHkA/647G8vnsTw0bE5KHbmdyaVen1veDodAiQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

다변수의 모델을 나타내면 위와 같다.  simple hyperplane은 초평면이라는 다차원 공간에서 나타나는 면이다. sign 함수는 시그모이드 함수이다. score 함수는 features 와 weights 에 대해 나타낸 함수이다. 

 

3. Class probabilities & Logistic regression
----------------------------------------------------

이제는 확률을 통해 분류를 시도한다. 2번째 모듈에서는 score 가 0을 기준으로 좋고 나쁨을 비교했다면 이번 모듈에서는 시그모이드 함수를 통해 확률로 표현할 것이다. 시그모이드 함수를 알아보기 전에 확률에 대한 이해가 필요하다. 

[![](https://1.bp.blogspot.com/-yuNlW7G23Ko/V5cKsdM3EDI/AAAAAAAAHkM/dR2sBHFgRdsqUIlKA2CaJvlMpy6-F8VfgCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-yuNlW7G23Ko/V5cKsdM3EDI/AAAAAAAAHkM/dR2sBHFgRdsqUIlKA2CaJvlMpy6-F8VfgCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

p( y=+1 | xi ) 라는 함수의 의미는 xi 가 주어졌을 때 y가 +1일 확률을 나타낸다. 그리고 p( y=+1 | xi )&nbsp;+&nbsp;p( y=-1 | xi ) = 1 이 성립한다. 이제 시그모이드 함수를 알아보자. 

[![](https://1.bp.blogspot.com/-qZR_a6Cf7yg/V5cLh2nkQgI/AAAAAAAAHkQ/kAWYd1oEuXwo71TNIyLcoRZzoHgZO7gEQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-qZR_a6Cf7yg/V5cLh2nkQgI/AAAAAAAAHkQ/kAWYd1oEuXwo71TNIyLcoRZzoHgZO7gEQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

score 함수의 값은 -무한대에서 무한대까지 존재할 수 있다. 그 범위를 아래와 같이 0부터 1까지로 만들어 확률로 만드려한다. 확률의 범위로 만들어주는 함수가 sign()인 시그모이드 함수이다. 시그모이드 함수는 아래와 같은 그래프를 갖는다. 

[![](https://4.bp.blogspot.com/-TwG1dub1ofA/V5cMNYgvp_I/AAAAAAAAHkY/2O9AtquYMgY40ndfe7QN1GZCtmVgWH7cwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://4.bp.blogspot.com/-TwG1dub1ofA/V5cMNYgvp_I/AAAAAAAAHkY/2O9AtquYMgY40ndfe7QN1GZCtmVgWH7cwCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

 위와 같이 무한대의 범위를 0부터 1까지의 범위로 만들어준다. y = +1 일 확률이 0.5 보다 크면 y hat = +1 이 된다.
 
[![](https://1.bp.blogspot.com/-WDUxGs3LfVc/V5cSNyrFuqI/AAAAAAAAHko/lsYOH73bLH4qMTF6WwmSFC2KXfjRN1AqwCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-WDUxGs3LfVc/V5cSNyrFuqI/AAAAAAAAHko/lsYOH73bLH4qMTF6WwmSFC2KXfjRN1AqwCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

 가중치들이 score에 주는 영향에 대해 위와 같이 나타낼 수 있다. 가중치가 커질수록 시그모이드 함수의 기울기는 커진다. 


4. Practical issues for classification
--------------------------------------
[![](https://1.bp.blogspot.com/-niKzeKm7Hvk/V5cV8Ztw3rI/AAAAAAAAHk0/g6Gu13N406c0yAic-Kg_Mbx1_4FfnL6eQCLcB/s400/%25EC%25BA%25A1%25EC%25B2%2598.PNG)](https://1.bp.blogspot.com/-niKzeKm7Hvk/V5cV8Ztw3rI/AAAAAAAAHk0/g6Gu13N406c0yAic-Kg_Mbx1_4FfnL6eQCLcB/s1600/%25EC%25BA%25A1%25EC%25B2%2598.PNG) 

위 그림은 다양한 feature들이 있을 때 분류할 경우(Mulitclass Classification)의 그래프이다. 이 때 1 versus all approach를 이용한다. 세모에 대한 분류를 구하고 싶을 때 경우의 수를 2가지로 압축한다. 세모인 경우와 세모가 아닌 경우로 나누어 간단히 하는 방법이다. 

 -------------------------------------------------------------------
> _**눈을 감아라.**_
> _**그럼 너는 너 자신을 볼 수 있으리라.**_ 
> **-버틀러-**