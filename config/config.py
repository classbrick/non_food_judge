


class config:
    ckpt_path = 'model/base_model/inception_resnet_v2_2016_08_30.ckpt'
    test_cpkt_path = 'D:/Users/zhengnan.luo/PycharmProjects/non_food_judge/model/save/gpu-nonfood_30000.000000_1.382898_0.906250/chamo.ckpt'
    output_path = 'model/'
    eval_tfrecord_path = 'E:/imagenet/imagenet/food_and_foodnon/20180706/tfrecord_test/'
    train_tfrecord_path = 'E:/imagenet/imagenet/food_and_foodnon/20180706/tfrecord_eval/'
    test_tfrecord_path = 'E:/imagenet/imagenet/food_and_foodnon/20180706/tfrecord_test/'
    model_save_path = 'model/output_model/'
    batchsize = 16
    image_width = 299
    image_height = 299
    class_num = 2

    def __init__(self):
        pass

