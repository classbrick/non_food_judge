import tensorflow as tf
import net.IRV2 as net
import os
from config import config
from preprocess import eval_preprocess
from preprocess import train_preprocess
slim = tf.contrib.slim


config_obj = config.config()
batchsize = config_obj.batchsize
image_width = config_obj.image_width
image_height = config_obj.image_height
ckpt = config_obj.ckpt_path

eval_tfrecord_path = config_obj.eval_tfrecord_path
train_tfrecord_path = config_obj.train_tfrecord_path

model_save_path = config_obj.model_save_path
class_num = config_obj.class_num

eval_preprocess_obj = eval_preprocess.eval_preprocess(eval_tfrecord_path, class_num)
train_preprocess_obj = train_preprocess.train_preprocess(train_tfrecord_path, class_num)

train_image, train_label = train_preprocess_obj.def_preposess(batchsize)
eval_image, eval_label = eval_preprocess_obj.def_preposess(batchsize)

output_train, _ = net.def_net(train_image, is_training=True)
output_eval, _ = net.def_net(eval_image, is_training=False)

saver = tf.train.Saver()

output_train = slim.fully_connected(output_train, class_num)
output_eval = slim.fully_connected(output_eval, class_num)

train_prediction = tf.nn.softmax(output_train)
eval_prediction = tf.nn.softmax(output_eval)


loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=output_train)
acc_train = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(train_prediction, 1), tf.arg_max(train_label, 1)), tf.float32))
acc_eval = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(eval_prediction, 1), tf.arg_max(eval_label, 1)), tf.float32))

optimizer = tf.train.AdamOptimizer(1e-4)
train_step = slim.learning.create_train_op(loss, optimizer)
loss = tf.reduce_sum(loss)


init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, save_path=ckpt)
    step = 0
    while True:
        _, loss_, acc_train_ = sess.run([train_step, loss, acc_train])
        step += 1
        print('[step:%d][loss_:%f][acc_train_:%f]' % (step, loss_, acc_train_))
        if step%100 == 0:
            acc_eval_ = sess.run([acc_eval])
            print('[step:%d][loss_:%f][acc_eval_:%f]' % (step, loss_, acc_eval_))
            if step > 5000 or acc_eval_ > 0.9:
                output_name = model_save_path + 'nonfood_%f_%f_%f' % (step, loss_, acc_eval_)
                os.system('mkdir ' + output_name)
                saver.save(sess, output_name + '/chamo.ckpt')
    coord.request_stop()
    coord.join(threads)

