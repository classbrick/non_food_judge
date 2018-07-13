import tensorflow as tf
import os
import numpy as np
from PIL import Image
slim = tf.contrib.slim


def convert_a_folder(tfrecord_root, folder_path, label):
    if not os.path.exists(tfrecord_root):
        os.makedirs(tfrecord_root)
    temp_folder_path = folder_path[0:len(folder_path)-2]
    folder_name = os.path.basename(temp_folder_path)
    print('converting %s' % folder_name)
    tfrecord_writer = tf.python_io.TFRecordWriter(tfrecord_root + folder_name + '.tfrecord')
    img_list = os.listdir(folder_path)
    for img_name in img_list:
        try:
            image_data = tf.gfile.FastGFile(folder_path + img_name, 'rb').read()
            img = Image.open(folder_path + img_name, 'r')
            size = img.size
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'label': tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_data])),
                    'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[0]])),
                    'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[size[1]]))
                }))
            tfrecord_writer.write(example.SerializeToString())
        except:
            continue
    tfrecord_writer.close()


def convert_folders(root_folder, tfrecord_root):
    '''
    将root_folder下的每一个子文件夹
    :param root_folder: 图片的位置，每个子文件夹包含一定量的图片
    :param tfrecord_root: 要存储tfrecord的位置
    :return:
    '''
    print('start convert')
    ret_dict = {}
    if not os.path.exists(root_folder):
        print('%s doesnt exist' % root_folder)
        return
    folder_list = os.listdir(root_folder)
    max_len = len(folder_list)
    for i in range(max_len):
        label = np.zeros([max_len], dtype=np.int64)
        label[i] = 1
        basename = folder_list[i]
        ret_dict[i] = basename
        folder = root_folder + basename + '/'
        convert_a_folder(tfrecord_root, folder, label)


if __name__ == '__main__':
    root_folder = 'E:/imagenet/imagenet/train_img/'
    tfrecord_root = 'E:/imagenet/imagenet/train_tfrecord/'
    convert_folders(root_folder, tfrecord_root)