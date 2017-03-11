1. Why use precision & recall as quality metrics
------

![](https://3.bp.blogspot.com/-ZbJwn2OfmZs/V6wdz6RdzAI/AAAAAAAAH1k/6ITAmu3ayPsMmnGZxVYjYSxYOiBMZWAbwCK4B/s400/ScreenShot_20160811153954.png)

정확도(accuracy) 측정 문제에서 정밀도(precision)과 재현율(recall)이 중요하다. 위 두가지는 trade off 관계로 이루어져 있다. 일반적으로 recall 이 높다면 precision이 낮아지고, recall 이 낮다면 precision이 높아진다.

2. Precision and recall explained
-------

![](https://2.bp.blogspot.com/-K8kTHAh_iCI/V6wlrpaBTmI/AAAAAAAAH2A/yDeF-RCFG1II3o4Ys4xuSFvrH5EXljF8QCK4B/s400/ScreenShot_20160811153954.png)

재현율과 정밀도를 설명하기 전에 내가 예측한 값과 실제 값에 대한 표를 기억해두자.

### 2.1 Precision

![](https://4.bp.blogspot.com/-OEAgtwnA9pI/V6wjM7dy4NI/AAAAAAAAH10/HkTE8DQR2joiVPq6pvo9UsBxPFpOftDrQCK4B/s400/ScreenShot_20160811153954.png)

precision을 위와 같이 그림으로 나타낼 수 있다. 내가 참이라고 예측한 값 중 실제 값이 참일 가능성이다.

#### precision = true positives / (true positives+ false positives)

최고값은 1.0 이고 최저값은 0.0 이다.

### 2.2 Recall

![](https://4.bp.blogspot.com/-041JyYzyDTY/V6wmzZ4MFtI/AAAAAAAAH2M/Tt5T9x1EqH0Fv4Q5FJsQ9CmfSFeO6cnMACK4B/s400/ScreenShot_20160811153954.png)

recall을 위와 같이 표현할 수 있다.
데이터 포인트의 모든 참 값 중 내가 참이라고 예측한 값일 가능성이다.
#### recall = true positives / (true positives+ false negative)
최고값은 1.0 이고 최저값은 0.0 이다.

3. The precision-recall trade-off
-----

![](https://3.bp.blogspot.com/-wOHxW5Bc5gA/V6wszR1rUiI/AAAAAAAAH2c/fwcVNKsqXvUBrQaO0p2pAsQB2JhDKG_eQCK4B/s400/ScreenShot_20160811153954.png)

우리가 원하는 모델은 높은 재현율과 낮은 정밀도를 갖는 모델이다. 즉, 모든 참인 데이터포인트 중 예측한 값이 참일 확률이 높은 모델이 좋은 모델이다.

![](https://3.bp.blogspot.com/-pJbwRbVWGFQ/V6wt4RuPhNI/AAAAAAAAH2k/FMtMdr-TJ0kkCeLgPBXAJ1XN5N3YIwNXQCK4B/s400/ScreenShot_20160811153954.png)

위와 같이 recall과 precision은 반비례관계를 갖고 있다.

![](https://2.bp.blogspot.com/-u0QF6mJvn5o/V6wulha1MiI/AAAAAAAAH2s/rCGiFvs4Ti4hSLzdvRIuoYpMBmbOomf3wCK4B/s400/ScreenShot_20160811153954.png)

그렇다면 어떻게 optimistic model 을 만들 수 있을까?
기본적인 방법은 threshold를 낮추는 것이다. 왜냐하면 너무 높이게 되면 실제 참인 값이 대부분 거짓값으로 예측하기 때문이다.

![](https://3.bp.blogspot.com/-Con8wcYs3XA/V6wvG3mtLRI/AAAAAAAAH20/44C0lODfp4IaNCjcCknoAWb1YjtqiVZ3wCK4B/s400/ScreenShot_20160811153954.png)

위 내용을 위 그림으로 요약할 수 있다. 이제는 classifier를 비교해보자.

![](https://3.bp.blogspot.com/-jMRYJ6F9d7I/V6wwDcZrvOI/AAAAAAAAH28/bhFbP1Jk-UghT3v33ZAPq6rTLhv0DHlsgCK4B/s400/ScreenShot_20160811153954.png)

같은 precision 값에서 어떤 classifier가 더 높은 recall 값을 갖는지 비교하면 어떤 classifier가 좋은 것인지 알 수 있다.

다음 모듈에서는 큰 데이터에서 왜 gradient descent 가 적용되기 힘든 지와 gradient descent를 대체할 알고리즘에 대해 알아볼 것이다. 또한 online streaming data 에 대해 간단히 알아 볼 것이다. 