import tensorflow as tf
import net.IRV2 as net
import numpy as np
from preprocess import eval_preprocess
from preprocess import train_preprocess
import utils.imageUtils as imageUtils
slim = tf.contrib.slim


batchsize = 16
image_width = 299
image_height = 299

ckpt = 'model/base_model/inception_resnet_v2_2016_08_30.ckpt'

eval_tfrecord_path = 'E:/imagenet/imagenet/eval_tfrecord/'
train_tfrecord_path = 'E:/imagenet/imagenet/train_tfrecord/'

eval_preprocess_obj = eval_preprocess.eval_preprocess(eval_tfrecord_path, 2)
train_preprocess_obj = train_preprocess.train_preprocess(train_tfrecord_path, 2)

train_image, train_label = train_preprocess_obj.def_preposess(batchsize)
eval_image, eval_lable = eval_preprocess_obj.def_preposess(batchsize)

output_train, _ = net.def_net(train_image, is_training=True)
output_eval, _ = net.def_net(eval_image, is_training=False)

saver = tf.train.Saver()

output_train = slim.fully_connected(output_train, 2)
output_eval = slim.fully_connected(output_eval, 2)

train_prediction = tf.nn.softmax(output_train)
eval_prediction = tf.nn.softmax(output_eval)


loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=output_train)
acc_train = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(train_prediction, 1), tf.arg_max(train_label, 1)), tf.float32))

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = slim.learning.create_train_op(loss, optimizer)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, save_path=ckpt)
    while True:
        _, loss_, acc_train_ = sess.run([train_step, loss, acc_train])
        print('[loss_:%f][acc_train_:%f]' % (loss_, acc_train_))
    coord.request_stop()
    coord.join(threads)

