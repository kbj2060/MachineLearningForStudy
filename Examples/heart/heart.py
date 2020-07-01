'''
 * llewyn 17.03.02
 * 3 Layers Neural Network using Stanford-Tutorial's heart.csv data
'''
import tensorflow as tf
import os

'''
 * Set up Paths ( Checkpoints, Summary, Data )
'''
SUMMARY_PATH = os.getcwd() + '/summary/'
CKPT_PATH = os.getcwd() + "/ckpt/heart.ckpt"
DATA_PATH = os.getcwd() + '/data/heart.csv'
DATA_LIST = [DATA_PATH]

'''
 * Set up Numbers
'''
FEATURE_NUM = 9
BATCH_SIZE = 50
CLASSES = 2
HIDDEN1_SIZE = 30
HIDDEN2_SIZE = 10
MAX_RANGE = 1000

'''
 * Get data using tensorflow and put it to queue
'''
data_queue = tf.train.string_input_producer(DATA_LIST)
reader = tf.TextLineReader(skip_header_lines=1)
_, value = reader.read(data_queue)

'''
 * Set up records for csv format
'''
record_defaults = [[1.0] for _ in range(FEATURE_NUM)]
record_defaults[4] = ['']
record_defaults.append([1])

'''
 * Change Strings to Numbers ( Present to 1.0 and Absent to 0.0 )
'''
content = tf.decode_csv(value, record_defaults=record_defaults)
condition = tf.equal(content[4], tf.constant('Present'))
content[4] = tf.select(condition, tf.constant(1.0), tf.constant(0.0))

'''
 * Pack the contents to features and Set label
'''
features = tf.pack(content[:FEATURE_NUM])
label = content[-1]

'''
 * Shuffle the feature batches and label batches
'''
feature_batch, label_batch = tf.train.shuffle_batch( [features, label], BATCH_SIZE, BATCH_SIZE * 100, BATCH_SIZE * 50)
feature_batch = tf.reshape(feature_batch, [BATCH_SIZE, FEATURE_NUM],name = "feature_batch")

'''
 * Reshape the label batch from 1 col to 2 cols using tf.one_hot ( ex) [1.0] => [1.0 , 0.0], [0.0] => [0.0 , 1.0] )
'''
label_batch = tf.reshape(label_batch, [BATCH_SIZE])
label_batch = tf.one_hot(label_batch, depth= CLASSES, dtype=tf.float32,name="label_batch")

'''
 * Hidden 1st Layer : Weights are initialized truncated_normal and shape is [FEATURE_NUM, HIDDEN1_SIZE]
 * Hidden 2nd Layer : Weights are initialized truncated_normal and shape is [HIDDEN1_SIZE, HIDDEN2_SIZE]
 * Output Layer : Weights are initialized truncated_normal and shape is [HIDDEN2_SIZE, CLASSES]
'''
w1 = tf.Variable(tf.truncated_normal([FEATURE_NUM, HIDDEN1_SIZE]), name="w1")
b1 = tf.Variable(tf.zeros([HIDDEN1_SIZE]), name="b1")

w2 = tf.Variable(tf.truncated_normal([HIDDEN1_SIZE, HIDDEN2_SIZE]), name="w2")
b2 = tf.Variable(tf.zeros([HIDDEN2_SIZE]), name="b2")

w3 = tf.Variable(tf.truncated_normal([HIDDEN2_SIZE, CLASSES]), name="w3")
b3 = tf.Variable(tf.zeros([CLASSES]), name="b3")

'''
 * Calculate with Activation functions ( sigmoid and softmax )
 * 1 Dropout between 1st and last layer.
'''
with tf.name_scope('hidden1') as h1_scope:
    h1 = tf.nn.sigmoid(tf.matmul(feature_batch, w1) + b1)

with tf.name_scope('hidden2') as h2_scope:
    h2 = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(h1, w2) + b2), keep_prob=0.8)

with tf.name_scope('output') as o_scope:
    y_ = tf.nn.softmax(tf.matmul(h2, w3) + b3)

'''
 * Cost : softmax_cross_entropy_with_logits
 * Optimizer : Adam
 * Evaluation : prediction y_ is rounded ( this is only for binary classification ) and compared to label
'''
with tf.name_scope('cost') as scope:
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits( tf.squeeze(y_), tf.squeeze(label_batch))

with tf.name_scope('train'):
    train = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

with tf.name_scope('evaluation'):
    correct_pred = tf.equal(tf.round(y_), label_batch)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float64))

'''
 * Saver : Save weights and bias
'''
param_list = [w1, b1, w2, b2, w3, b3]
saver = tf.train.Saver(param_list)

'''
 * Seesion open!
'''
with tf.Session() as sess:
    '''
     * threads manager 'coord' and queue runner 'threads' are initialized because data has stored in queue using shuffle
    '''
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    '''
     * merge all summaries and write summaries
     * init all variables
    '''
    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(SUMMARY_PATH, sess.graph)
    sess.run(tf.initialize_all_variables())

    '''
     * training and cheking accuracy until MAX_RANGE times if there is no data in queue , stop the iterations
    '''
    for step in range(MAX_RANGE):
        if coord.should_stop():
            break
        _, summ, acc = sess.run([train, merged, accuracy])
        if step % 100 == 0:
            saver.save(sess, CKPT_PATH)
            train_writer.add_summary(summ, step)
            print acc

    '''
     * stop the threads and exit
    '''
    coord.request_stop()
    coord.join(threads)