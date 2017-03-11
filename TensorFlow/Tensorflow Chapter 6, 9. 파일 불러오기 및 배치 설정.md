Chapter 6, 10. 파일 불러오기 및 배치 설정
===================
설명에 앞서 코드는 아래와 같다.

```python
import tensorflow as tf
from PIL import Image
import os

IMAGEDIR_PATH = os.getcwd() + '/../data/Test_Dataset_png/'
IMAGES = os.listdir( IMAGEDIR_PATH )
IMAGES.sort()
IMAGE_LIST = [ IMAGEDIR_PATH + name for name in IMAGES ]
LABEL_PATH = os.getcwd() + '/../data/Test_Dataset_csv/Label.csv'
LABEL_LIST = [ LABEL_PATH ]

image_width = 49
image_height = 61

imagename_queue = tf.train.string_input_producer( IMAGE_LIST )
labelname_queue = tf.train.string_input_producer( LABEL_LIST )

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read( imagename_queue )
label_key, label_value = label_reader.read( labelname_queue )

image_decoded = tf.cast( tf.image.decode_png( image_value ), tf.float32 )
label_decoded = tf.cast( tf.decode_csv( label_value, record_defaults=[[0]]), tf.float32 )

label = tf.reshape( label_decoded, [1] )
image = tf.reshape( image_decoded, [image_width, image_height, 1] )

x, y_= tf.train.shuffle_batch( tensors = [image, label], batch_size = 32, num_threads = 4, capacity = 5000, min_after_dequeue = 100 )

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners( sess = sess, coord = coord )

    try:
        while not coord.should_stop():
            print "image  : ", sess.run(x)
            print "label : ", sess.run(y_)
    except tf.errors.OutOfRangeError:
        print 'Done searching -- epoch limit reached'
    finally:
        coord.request_stop()

    coord.join(thread)
```

텐서플로우는 파일을 큐에 넣어 놓고 세션에서 하나씩 빼내서 파일을 처리한다. 굉장히 생소한 개념이고 처음 보고 이해가 잘 가지 않았다. 도대체 왜 shuffle_batch 함수에서 return된 텐서들은 세션 안에서 미니 배치를 갖고 돌 수 있는 것인가? 나중엔 그냥 너무 깊게 이해하지 않으려 했다. 파일 불러오기부터 설명하자면 코드는 아래와 같다.

#### 1. 파일 불러오기

```python
IMAGE_LIST = [ IMAGEDIR_PATH + name for name in IMAGES ]
imagename_queue = tf.train.string_input_producer( IMAGE_LIST )
```

우선 불러오고자 하는 이미지들의 경로와 이미지 이름을 합쳐 list에 넣어 tf.train.string_input_producer 함수에 넣는다. 결과 문자열을 큐에 넣어 리턴하는 함수이다. 이 큐는 session에서 QueueRunner로 하나씩 꺼내 쓸 수 있다. 

```python
image_reader = tf.WholeFileReader()
image_key, image_value = image_reader.read( imagename_queue )
```
reader 객체를 선언하고 큐에 넣은 이미지들을 read하는 함수를 이용한다. 그 리턴값은 key와 value가 나온다. key는 파일의 경로를 나타내고, value는 파일이 나타내는 값을 나타낸다.  

```python
image_decoded = tf.cast( tf.image.decode_png( image_value ), tf.float32 )
```
이미지 파일을 행렬로 바꾸는 함수이다. decoding 함수와 cast 함수를 이용해 str을 float으로 바꾸어 행렬로 표현한다.


#### 2. 미니 배치 및 큐

```python
x, y_= tf.train.shuffle_batch( tensors = [image, label], batch_size = 32, num_threads = 4, capacity = 5000, min_after_dequeue = 100 )
```

MNIST 예제를 보면 next_batch 사용자 정의 함수를 만들어 배치를 for 문으로 돌려 다음 배치를 트레이닝 시키지만 tensorflow 는 shuffle_batch를 지원한다. 세션에서 큐에 데이터가 없어질 때까지 while문을 돌려 배치를 꺼낸다. 

```python
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners( sess = sess, coord = coord )
```
그래프를 모두 그렸다면 세션 안에서 스레드를 관리하는 Coordinator 객체를 선언하고 queue runner를 통해 스레드를 실행시킨다.

```python
    try:
        while not coord.should_stop():
            print "image  : ", sess.run(x)
            print "label : ", sess.run(y_)
    except tf.errors.OutOfRangeError:
        print 'Done searching -- epoch limit reached'
    finally:
        coord.request_stop()
```

배치를 이용하는 부분이다. 반복문 안에서 큐를 관리하는 Coordinator의 should_stop 함수가 true 값이 나올 때(스레드를 멈춰야할 때)까지 x, y_ 값을 출력한다. 위에 선언한 32개의 한 배치가 출력되는 과정이 큐에 데이터가 없을 때까지 반복된다. 마지막에는 coord.request_stop 함수를 통해 스레드를 멈추도록 한다.









