import tensorflow as tf
import tf_utils as U
import global_var as gl
import time

class Net(object):
    def __init__(self, scope, globalMadan=None):
        raise NotImplemented

    #更新tensorflow中的权重和偏置
    def update_tf_wb(self, params, neulist):
        return params

    #更新全局网络的参数
    def update_global(self):
        raise NotImplemented

    #更新缓存中的参数
    def update_cache(self):
        raise NotImplemented

    #拉取全局网络的参数
    def pull_global(self):
        raise NotImplemented

    #向上推送本地参数
    def push_local(self):
        raise NotImplemented

    #更新神经元列表
    def update_neulist(self):
        raise NotImplemented

    #更新信息素
    def update_ph(self):
        raise NotImplemented

    #选择用于梯度下降的loss值
    def choose_loss(self, loss):
        return loss

    #计算回报
    def get_reward(self,loss):
        reward = loss
        return loss

    #保存模型
    def save_modle(self):
        raise NotImplemented

    #加载模型
    def load_modle(self):
        raise NotImplemented

    #测试训练结果
    def predict(self):
        raise NotImplemented


class trainer(Net):
    def __init__(self,name, arglist, X_input, Y_true):
        self.name = name
        self.arglist = arglist
        self.X_input = X_input
        self.Y_true = Y_true

    # 建立网络模型
    def net_model(self, X_input):
        out, w1, b1 = U.add_layer(X_input, X_input.shape[0], 5, activation_function=tf.nn.relu)
        out, w2, b2 = U.add_layer(out, 5, 8, activation_function=tf.nn.relu)
        out, w3, b3 = U.add_layer(out, 8, 1, activation_function=tf.nn.sigmoid)
        params = {'w': [w1, w2, w3],
                  'b': [b1, b2, b3]}
        return out, params

    #worker的训练过程
    def train(self):
        xs = tf.placeholder(tf.float32, [None, 1])
        ys = tf.placeholder(tf.float32, [None, 1])
        NN_out, params = self.net_model(self.X_input)
        self.update_tf_wb(params, 'neulist')
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.Y_true - NN_out), keepdims=True))
        reward = self.get_reward(loss)
        loss = self.choose_loss(loss)
        train_step = tf.train.GradientDescentOptimizer(0.001). minimize(loss)

        sess = tf.Session(U.make_session())
        sess.run(tf.global_variables_initializer())

        for i in range(20):
            start = time.time()
            sess.run(train_step, feed_dict={xs:self.X_input, ys:self.Y_true})
            end = time.time()

            #计算训练一次的时间
            step_time = start-end

            #消除不用更新的权重和偏置
            self.update_tf_wb(params, neulist=None)

            #推送本地参数到缓存区
            self.push_local()

            #拉取全局参数
            self.pull_global()


            print('step time:%f' % step_time)

            if i % 50 == 0:
                print(sess.run(loss, feed_dict={self.X_input:self.X_input, self.Y_true:self.Y_true}))

            if 'need saving model':
                self.save_modle()

class predictor(Net):
    def __init__(self):
        raise NotImplemented

    def predict(self, X_input):
        self.load_modle()

        result = []
        return result