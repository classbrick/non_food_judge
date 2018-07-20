import tensorflow as tf
import net.IRV2 as net
import os
import shutil
import preprocess.direct.imageUtils as imageUtils
import result_eval.default_eval as default_eval
from config import config
from preprocess.tfrecord import eval_preprocess

slim = tf.contrib.slim


def eval_tfrecord():
    config_obj = config.config()
    batchsize = config_obj.batchsize
    image_width = config_obj.image_width
    image_height = config_obj.image_height
    ckpt = config_obj.test_cpkt_path

    test_tfrecord_path = config_obj.test_tfrecord_path

    model_save_path = config_obj.model_save_path
    class_num = config_obj.class_num

    test_tfrecord_path = eval_preprocess.eval_preprocess(test_tfrecord_path, class_num)
    test_image, label = test_tfrecord_path.def_preposess(batchsize)
    output, _ = net.def_net(test_image, is_training=False)
    output = slim.fully_connected(output, class_num, scope='add_fully_connected')
    output = tf.nn.softmax(output)
    output_cast = tf.cast(output > 0.5, tf.float32)

    eval_list = default_eval.run_eval_and_save(output_cast, label, '')
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, save_path=ckpt)
        step = 100
        acc = 0.0
        pre_nume = 0.0
        pre_deno = 0.0
        rec_nume = 0.0
        rec_deno = 0.0
        for i in range(step):
            output_, output_cast_, eval_list_ = sess.run([output, output_cast, eval_list])
            acc += eval_list_[0]
            pre_nume += eval_list_[3]
            pre_deno += eval_list_[4]
            rec_nume += eval_list_[5]
            rec_deno += eval_list_[6]
            print(
                '[step:%d][acc:%f][pre:%f][rec:%f]' % (i + 1, acc / (i + 1), pre_nume / pre_deno, rec_nume / rec_deno))
            # print('output:', output_)
            # print('output_cast:', output_cast_)
        acc = acc / step
        pre = pre_nume / pre_deno
        rec = rec_nume / rec_deno
        print('%d eval steps done, totally evaluated %d pictures.' % (step, step * batchsize))
        print('[acc:%f][pre:%f][rec:%f]' % (acc, pre, rec))
        # while True:
        #     _, loss_, acc_train_ = sess.run([train_step, loss, acc_train])
        #     step += 1
        #     print('[step:%d][loss_:%f][acc_train_:%f]' % (step, loss_, acc_train_))
        #     if step%5 == 0:
        #         acc_eval_ = sess.run([acc_eval])
        #         print('[step:%d][loss_:%f][acc_eval_:%f]' % (step, loss_, acc_eval_[0]))
        #         if step > 5000 or acc_eval_[0] > 0.9:
        #             output_name = model_save_path + 'nonfood_%f_%f_%f' % (step, loss_, acc_eval_[0])
        #             os.system('mkdir ' + output_name)
        #             saver.save(sess, output_name + '/chamo.ckpt')
        coord.request_stop()
        coord.join(threads)


def eval_single_pic(pic_root_path, output_path):
    '''
    遍历pic_root_path下的所有图片(不递归)，然后菜[1,0]存入output_path/food，非菜[0,1]存入output_path/nonfood
    :param pic_path:
    :param output_path:
    :return:
    '''
    if not os.path.exists(pic_root_path):
        print('%s doesnt exist.' % pic_root_path)
        return
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    foodpath = output_path + 'food/'
    nonfoodpath = output_path + 'nonfood/'
    if not os.path.exists(foodpath):
        os.mkdir(foodpath)
    if not os.path.exists(nonfoodpath):
        os.mkdir(nonfoodpath)
    config_obj = config.config()
    image_width = config_obj.image_width
    image_height = config_obj.image_height
    ckpt = config_obj.test_cpkt_path
    class_num = config_obj.class_num
    image_p = tf.placeholder(dtype=tf.float32, shape=[1, image_width, image_height, 3])

    output, _ = net.def_net(image_p, is_training=False)
    output = slim.fully_connected(output, class_num, scope='add_fully_connected')
    output = tf.nn.softmax(output)
    output = tf.cast(output > 0.5, tf.float32)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        saver.restore(sess, save_path=ckpt)

        pic_list = os.listdir(pic_root_path)
        print('total pics: ', len(pic_list))
        food = 0
        nonfood = 0
        for pic_path in pic_list:
            pic_path = os.path.join(pic_root_path, pic_path)
            print('judging: %s', pic_path)
            image = imageUtils.read_a_pic_reconstruct_slim(pic_path, image_height, image_width)
            try:
                result = sess.run(output, feed_dict={image_p: image})
            except:
                continue
            print('pic_path: ' + pic_path)
            print('the result is: ' + str(result))
            # 这里进行输出[1, 0]代表菜，[0, 1]代表非菜
            if result[0][0] > 0.5:
                shutil.copy(pic_path, foodpath)
                food += 1
            else:
                shutil.copy(pic_path, nonfoodpath)
                nonfood += 1
        print('food: ', food)
        print('nonfood: ', nonfood)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    eval_tfrecord()
    # eval_single_pic('E:/food_imagenet_food/food/', 'E:/food_imagenet_food/my_pc_20000/')