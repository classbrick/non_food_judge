import net.InceptionResnetV2.InceptionResnetV2 as irv2
import tensorflow as tf
slim = tf.contrib.slim


def inception_arg_scope():
    return None


def def_net(inputs, num_classes=1001, is_training=True):
    with slim.arg_scope(irv2.inception_resnet_v2_arg_scope()):
        logits, endpoints = irv2.inception_resnet_v2(inputs, num_classes=num_classes, is_training=is_training)
        return logits, endpoints