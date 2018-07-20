import tensorflow as tf


def run_eval_and_save(prediction, label, output_path):
    '''
    输出acc、pre、label
    :param prediction:
    :param label:
    :param output_path:
    :return:
    '''
    prediction = tf.slice(prediction, [0, 0], [-1, 1])
    label = tf.slice(label, [0, 0], [-1, 1])
    acc = tf.reduce_mean(tf.cast(tf.equal(prediction, label), tf.float32))
    pre_nume = tf.reduce_sum(prediction) - tf.reduce_sum(tf.cast((prediction - label) >= 0.5, tf.float32))  # 精准率分子
    pre_deno = tf.reduce_sum(prediction)  # 精准率分母
    rec_nume = tf.reduce_sum(label) - tf.reduce_sum(tf.cast((label - prediction) >= 0.5, tf.float32))  # 召回分子
    rec_deno = tf.reduce_sum(label)  # 召回分母
    pre = pre_nume/pre_deno
    rec = rec_nume/rec_deno
    return [acc, pre, rec, pre_nume, pre_deno, rec_nume, rec_deno]
