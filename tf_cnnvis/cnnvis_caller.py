import os
import time
import tf_cnnvis.tf_cnnvis as tf_cnnvis

def show_deconv(sess, feed_dict, input_tensor, logpath, outpath, layers = []):
    '''
    deconv visualization
    :param sess: the sess
    :param feed_dict:
    :param input_tensor:
    :param logpath:
    :param outpath:
    :return:
    '''
    layers = ["r", "p", "c"]
    total_time = 0

    start = time.time()
    is_success = tf_cnnvis.deconv_visualization(
        sess_graph_path=sess, value_feed_dict=feed_dict,
        input_tensor=input_tensor, layers=layers,
        path_logdir=logpath,
        path_outdir=outpath)
    start = time.time() - start
    print("Total Time = %f" % (start))