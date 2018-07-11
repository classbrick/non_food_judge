import tensorflow as tf
import net.IRV2 as net
import numpy as np
import utils.imageUtils as imageUtils

batchsize = 1
image_width = 299
image_height = 299

ckpt = 'model/inception_resnet_v2_2016_08_30.ckpt'
pic_path = 'E:/test_data/myevalpics/pics1/2.jpg'

image_p = tf.placeholder(shape=[batchsize, image_width, image_height, 3], dtype=np.float32)
label_p = tf.placeholder(shape=[batchsize], dtype=np.float32)

output, _ = net.def_net(image_p, is_training=False)

saver = tf.train.Saver()

images, labels = imageUtils.read_a_pic(pic_path, dim=2, imagewidth=image_width, imageheight=image_height)


with tf.Session() as sess:
    saver.restore(sess, save_path=ckpt)
    output_ = sess.run(output, feed_dict={image_p: images})
    print(output_)

