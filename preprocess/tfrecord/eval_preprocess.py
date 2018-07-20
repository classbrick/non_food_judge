import tensorflow as tf
import utils.data_helper
import utils.global_var


def preprocess_for_train(image,
                         output_height,
                         output_width):
    image = utils.data_helper._aspect_preserving_resize(image, utils.global_var._RESIZE_SIDE_MIN)
    image = utils.data_helper._central_crop([image], output_height, output_width)[0]
    image.set_shape([output_height, output_width, 3])
    image = tf.to_float(image)
    return utils.data_helper._mean_image_subtraction(image)


class eval_preprocess:
    tfrecord_addr=None
    class_num = None
    def __init__(self,tfrecord_addr, class_num):
        print('choose eval_preprocess')
        self.tfrecord_addr =tfrecord_addr
        self.class_num = class_num
    def def_preposess(self, batch_size):
        image, label = utils.data_helper.get_raw_img(self.tfrecord_addr, self.class_num)
        train_image_size = utils.global_var._RESIZE_SIDE_MIN
        image = preprocess_for_train(image, train_image_size, train_image_size)
        c = batch_size
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=c,
            num_threads=3,
            capacity=200 * c,
            min_after_dequeue=199 * c
        )
        return images, labels
