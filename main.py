import tensorflow as tf
import net.IRV2 as net
import os
import time
from config import config
from preprocess.tfrecord import train_preprocess, eval_preprocess
slim = tf.contrib.slim

try:
  image_summary = tf.image_summary
  scalar_summary = tf.scalar_summary
  histogram_summary = tf.histogram_summary
  merge_summary = tf.merge_summary
  SummaryWriter = tf.train.SummaryWriter
except:
  image_summary = tf.summary.image
  scalar_summary = tf.summary.scalar
  histogram_summary = tf.summary.histogram
  merge_summary = tf.summary.merge
  SummaryWriter = tf.summary.FileWriter


def fully_connected(input, class_num, scopename='add_fully_connected'):
    output = slim.fully_connected(input, class_num, scope=scopename, reuse=tf.AUTO_REUSE)
    return output





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

output_train = fully_connected(output_train, class_num)
output_eval = fully_connected(output_eval, class_num)

train_prediction = tf.nn.softmax(output_train)
eval_prediction = tf.nn.softmax(output_eval)


loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_label, logits=output_train)
# temp = tf.arg_max(train_label, 1)
acc_train = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(train_prediction, 1), tf.arg_max(train_label, 1)), tf.float32))
acc_eval = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(eval_prediction, 1), tf.arg_max(eval_label, 1)), tf.float32))

acc_summary_train = scalar_summary('accuracy_train', acc_train)
acc_summary_eval = scalar_summary('accuracy_eval', acc_train)


optimizer = tf.train.AdamOptimizer(1e-4)
train_step = slim.learning.create_train_op(loss, optimizer)
loss = tf.reduce_sum(loss)
loss_summary = scalar_summary('loss', loss)

init = tf.initialize_all_variables()

saver_all = tf.train.Saver(max_to_keep=360)


merged = merge_summary([loss_summary, acc_summary_eval, acc_summary_train])
with tf.Session() as sess:
    sess.run(init)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    saver.restore(sess, save_path=ckpt)
    writer = SummaryWriter('save-cnn20/logs', sess.graph)
    step = 0
    while True:
        start_time = time.time()
        summary_, _, loss_, acc_train_ = sess.run([merged, train_step, loss, acc_train])
        step += 1
        writer.add_summary(summary_, step)
        print('[step:%d][loss_:%f][acc_train_:%f][time_cost:%f]' % (step, loss_, acc_train_, time.time()-start_time))
        # temp_ = sess.run(temp)
        # print(temp_)
        if step%5 == 0:
            acc_eval_ = sess.run([acc_eval])
            print('[step:%d][loss_:%f][acc_eval_:%f]' % (step, loss_, acc_eval_[0]))
            if step % 50 == 0:
                output_name = model_save_path + 'nonfood_%f_%f_%f' % (step, loss_, acc_eval_[0])
                os.system('mkdir ' + output_name)
                saver_all.save(sess, output_name + '/chamo.ckpt')
            # saver_all.save(sess, output_name, global_step=step)
    coord.request_stop()
    coord.join(threads)

