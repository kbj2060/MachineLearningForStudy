'''
 * llewyn 17.03.09
 * 4 Layers Neural Network using restaurant sales data
'''
import tensorflow as tf
import os

'''
 * Set up Paths and Numbers
'''
CKPT_PATH = os.getcwd() + "/ckpt/sales.ckpt"
SUMMARY_PATH = os.getcwd() + '/summary/'
FILE_PATH = 'data.csv'
FEATURE_NUM = 3
LABEL_NUM = 1
BATCH_SIZE = 150
HIDDEN1_SIZE = 20
HIDDEN2_SIZE = 50
HIDDEN3_SIZE = 30
CLASSES = 4
MAX_RANGE = 100000

'''
 * Get data using tensorflow and put it to queue
'''
filename_queue = tf.train.string_input_producer([FILE_PATH])
reader = tf.TextLineReader()
_, value = reader.read(filename_queue)

'''
 * Set up records for csv format
'''
record_defaults = [ [1.0] for _ in range(FEATURE_NUM + LABEL_NUM) ]
content = tf.decode_csv(value, record_defaults=record_defaults)

'''
 * Pack the contents to features and Set label
'''
features = tf.pack(content[1:])
label = content[0]

'''
 * Shuffle the feature batches and label batches
'''
feature_batch, label_batch = tf.train.shuffle_batch( [features, label], BATCH_SIZE, BATCH_SIZE * 100, BATCH_SIZE * 50)

'''
 * Reshape the label batch from 1 col to 4 cols using tf.one_hot ( ex) [0.0] => [1.0 , 0.0, 0.0, 0.0] )
'''
label_batch = tf.cast(tf.reshape(label_batch, [BATCH_SIZE]), tf.int32)
label_batch = tf.one_hot(label_batch, depth= CLASSES, on_value = 1, off_value = 0, name="label_batch")

'''
 * Input 1st Layer : Weights are initialized truncated_normal and shape is [FEATURE_NUM, HIDDEN1_SIZE]
 * Hidden 2nd Layer : Weights are initialized truncated_normal and shape is [HIDDEN1_SIZE, HIDDEN2_SIZE]
 * Hidden 3rd Layer : Weights are initialized truncated_normal and shape is [HIDDEN2_SIZE, HIDDEN3_SIZE]
 * Output Layer : Weights are initialized truncated_normal and shape is [HIDDEN3_SIZE, CLASSES]
'''
w1 = tf.Variable(tf.truncated_normal([FEATURE_NUM, HIDDEN1_SIZE]), name="w1")
b1 = tf.Variable(tf.zeros([HIDDEN1_SIZE]), name="b1")

w2 = tf.Variable(tf.truncated_normal([HIDDEN1_SIZE, HIDDEN2_SIZE]), name="w2")
b2 = tf.Variable(tf.zeros([HIDDEN2_SIZE]), name="b2")

w3 = tf.Variable(tf.truncated_normal([HIDDEN2_SIZE, HIDDEN3_SIZE]), name="w3")
b3 = tf.Variable(tf.zeros([HIDDEN3_SIZE]), name="b3")

w4 = tf.Variable(tf.truncated_normal([HIDDEN3_SIZE, CLASSES]), name="w4")
b4 = tf.Variable(tf.zeros([CLASSES]), name="b4")

'''
 * Calculate with Activation functions ( elu, elu and softmax )
 * There are regularization r2 ( l2_loss )
 * #There is a 0.6 dropout in hidden3
'''
with tf.name_scope('input') as h1_scope:
    h1 = tf.nn.elu(tf.matmul(feature_batch, w1) + b1)

with tf.name_scope('hidden2') as h2_scope:
    h2 = tf.nn.elu(tf.matmul(h1, w2) + b2)

with tf.name_scope('hidden3') as h3_scope:
    h3 = tf.nn.elu(tf.matmul(h2, w3) + b3)

with tf.name_scope('output') as o_scope:
    y_ = tf.nn.softmax(tf.matmul(h3, w4) + b4)

'''
 * Cost : softmax_cross_entropy_with_logits
 * Optimizer : Adam
 * Evaluation : Compare the index, which has biggest probability, of prediction y_ and label
'''
with tf.name_scope('cost') as scope:
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_, tf.cast(label_batch, tf.float32))+
                                                                                        0.01 * tf.nn.l2_loss(w1)+
                                                                                        0.01 * tf.nn.l2_loss(w2)+
                                                                                        0.01 * tf.nn.l2_loss(w3))
with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
with tf.name_scope('evaluation'):
    correct_pred = tf.equal(tf.argmax(y_, 1), tf.argmax(label_batch, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

'''
 * Saver : Save weights and bias
'''
param_list = [w1, b1, w2, b2, w3, b3, w4, b4]
saver = tf.train.Saver(param_list)

'''
 * Seesion open!
'''
with tf.Session() as sess:
    '''
     * threads manager 'coord' and queue runner 'threads' are initialized because data has stored in queue using shuffle
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    '''
     * merge all summaries and write summaries
     * init all variables
    '''
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(SUMMARY_PATH, sess.graph)
    sess.run(tf.global_variables_initializer())

    '''
     * training and cheking accuracy until MAX_RANGE times if there is no data in queue , stop the iterations
    '''
    for step in range(MAX_RANGE):
        if coord.should_stop():
            break
        _, merge, cost, acc = sess.run([train, merged, cross_entropy, accuracy])
        if step % 100 == 0:
            saver.save(sess, CKPT_PATH)
            train_writer.add_summary(merge, step)
            print "step : ",step," cost : ",cost," acc : ",acc

    '''
     * stop the threads and exit
    '''
    coord.request_stop()
    coord.join(threads)