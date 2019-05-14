import tensorflow as tf

#添加网络层
def add_layer(input,in_size,out_size,activation_function=None):
    W = tf.Variable(tf.random_normal([in_size, out_size]), dtype=tf.float32)
    b = tf.Variable((tf.zeros([1, out_size]) + 0.1), dtype=tf.float32)
    Z = tf.matmul(input, W) + b
    # 根据是否有激活函数
    if activation_function == None:
        output = Z
    else:
        output = activation_function(Z)
    return output, W, b

#使用cpu开核数量 或 使用gpu
def make_session():
    cpu_num = int
    tf_config = None
    return tf_config
