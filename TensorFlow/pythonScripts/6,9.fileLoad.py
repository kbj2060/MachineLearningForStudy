import tensorflow as tf
from PIL import Image
import os

IMAGEDIR_PATH = os.getcwd()+'/../data/Test_Dataset_png/'
IMAGES = os.listdir(IMAGEDIR_PATH)
IMAGES.sort()
IMAGE_LIST = [IMAGEDIR_PATH + name for name in IMAGES ]
LABEL_PATH = os.getcwd()+ '/../data/Test_Dataset_csv/Label.csv'
LABEL_LIST = [ LABEL_PATH ]

image_width = 49
image_height = 61
imagename_queue = tf.train.string_input_producer(IMAGE_LIST)
labelname_queue = tf.train.string_input_producer(LABEL_LIST)

image_reader = tf.WholeFileReader()
label_reader = tf.TextLineReader()

image_key, image_value = image_reader.read(imagename_queue)
label_key, label_value = label_reader.read(labelname_queue)

image_decoded = tf.cast(tf.image.decode_png(image_value), tf.float32)
label_decoded = tf.cast(tf.decode_csv(label_value, record_defaults=[[0]]), tf.float32)

label = tf.reshape(label_decoded, [1])
image = tf.reshape(image_decoded,[image_width,image_height,1])
image1 = tf.squeeze(image_decoded)

x, y_= tf.train.shuffle_batch(tensors=[image, label], batch_size=32, num_threads=4, capacity=5000, min_after_dequeue=100)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    thread = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        while not coord.should_stop():
            print "image  : ", sess.run(x)
            print "label  : ", sess.run(y_)
    except tf.errors.OutOfRangeError:
        print('Done searching -- epoch limit reached')
    finally:
        coord.request_stop()
    coord.request_stop()
    coord.join(thread)
