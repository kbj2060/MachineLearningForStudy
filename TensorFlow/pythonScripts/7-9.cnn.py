import tensorflow as tf
from PIL import Image

IMAGE_PATH = '../data/Face00001.png'
LABEL_PATH = '../data/Label.csv'
IMAGE_LIST = [ IMAGE_PATH ]
LABEL_LIST = [ LABEL_PATH ]
IMAGE_WIDTH = 49
IMAGE_HEIGHT = 61

label_queue = tf.train.string_input_producer(LABEL_LIST)
image_queue = tf.train.string_input_producer(IMAGE_LIST)

reader_csv = tf.TextLineReader()
reader_image = tf.WholeFileReader()

label_key, label_value = reader_csv.read(label_queue)
key, value = reader_image.read(image_queue)

image_decoded = tf.image.decode_png(value)
label_decoded = tf.decode_csv(label_value,record_defaults = [[0]])

x = tf.cast(image_decoded, tf.float32)
y_ = tf.cast(label_decoded, tf.float32)
y_ = tf.reshape(y_, [-1,1])

w1 = tf.Variable(tf.truncated_normal([5,5,1,32]))
b1 = tf.Variable(tf.zeros([32]))
x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT,1])
conv1 = tf.nn.conv2d(x_image, w1, strides=[1,1,1,1], padding = 'SAME')
h1 = tf.nn.relu(conv1 + b1)

w2 = tf.Variable(tf.truncated_normal([5,5,32,64]))
b2 = tf.Variable(tf.truncated_normal([64]))

conv2 = tf.nn.conv2d(h1, w2, strides=[1,1,1,1], padding="SAME")
h2 = tf.nn.relu(conv2 + b2)

h_flat = tf.reshape(h2, [-1,49*61*64])
fc_w = tf.Variable(tf.truncated_normal([49*61*64, 10]))
fc_b = tf.Variable(tf.zeros([1]))
fc1 = tf.nn.relu(tf.matmul(h_flat, fc_w) + fc_b)
drop_fc1 = tf.nn.dropout(fc1, 0.5)

W_out = tf.Variable(tf.truncated_normal([10,1]))
b_out = tf.Variable(tf.zeros([1]))

pred = tf.matmul(drop_fc1, W_out) + b_out
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pred, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
correct_prediction =  tf.equal(tf.argmax(pred,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
	init = tf.initialize_all_variables()
	sess.run(init)
	coord = tf.train.Coordinator()
	thread = tf.train.start_queue_runners(sess=sess,coord=coord)
		
	for i in range(10):	
		_, cost, acc = sess.run([train_step,loss, accuracy])
		print cost
		print acc
    
	coord.request_stop()
	coord.join(thread)
