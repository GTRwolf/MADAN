import tensorflow as tf
import tf_utils as U
import train as T
import threading
import global_var as gl
import numpy as np

global param_cache

#加载数据
def load_data():
    X = tf.constant([1.05])
    Y = tf.constant([2.10])
    return X, Y

#参数结束
def parse_args():
    arglist = []
    return arglist

#主函数
if __name__ == "__main__":

    #初始化全局变量，建立用于存储全局的训练信息缓存
    gl._init()

    #加载训练数据
    X, Y = load_data()

    #解析参数
    arglist = parse_args()

    #设置使用cpu开核数量 或 使用gpu
    sess = tf.Session(config=U.make_session())
    with tf.device("/cpu:0"):
        trainers = []
        # 创建worker
        for i in range(Y.shape[0]):
            i_name = 'w_%i' % i
            trainers.append(T.trainer(i_name, arglist, X,Y))

    #加入线程协调器
    COORD = tf.train.Coordinator()

    # 调用work　开始训练
    trainer_threads = []

    for trainer in trainers:
        job = lambda : trainer.train()
        t = threading.Thread(target= job)
        t.start()
        trainer_threads.append(t)
    COORD.join(trainer_threads)

