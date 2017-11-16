import tensorflow as tf
import re,datetime,os
import TFRecord
TOWER_NAME = 'tower'

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999        # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 1000.0        # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1     # Learning rate decay factor.
INITIAL_LEARNING_RATE = 1e-4         # Initial learning rate.
LOG_FREQUENCY = 10                   # 多少步控制台打印一次结果
SESS_CONFIG = tf.ConfigProto(allow_soft_placement=True,
                             log_device_placement=True,
                             gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
initializer_conv2d = tf.contrib.layers.xavier_initializer_conv2d()
initializer_fully = tf.contrib.layers.xavier_initializer()

class CNNClassify(object):
    """CNN图像分类
    """
    def __init__(self, batch_size, num_classes, num_train_examples,session_conf=SESS_CONFIG):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_train_examples = num_train_examples  # 训练样本数量
        self.session_conf = session_conf
        self.max_steps = 100000
        self.checkpoint_every = 1000


    def _variable_on_cpu(self, name, shape, initializer):
        """帮助创建存储在CPU内存上的变量。"""
        with tf.device('/cpu:0'):
            var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
        return var


    def _variable_with_weight_decay(self, name, shape, initializer, wd=None):
        """初始化权重变量
        Args:
          name: name of the variable
          shape: list of ints
          stddev: 高斯函数标准差
          wd: 添加L2范数损失权重衰减系数。如果没有，该变量不添加重量衰减。
        Returns:权重变量
        """
        var = self._variable_on_cpu(name, shape, initializer)
        if wd is not None:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def _activation_summary(self, x):
        """创建tensorboard摘要 好可视化查看
        """
        tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
        tf.summary.histogram(tensor_name + '/activations', x)
        tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def conv2d_layer(self,name,input,w_shape,initializer,b_shape,wd=None):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay('weights', shape=w_shape, initializer=initializer)  # 权值矩阵
            # 二维卷积
            conv = tf.nn.conv2d(input, kernel, [1, 1, 1, 1], padding='SAME')   #  周围补0 保持形状不变
            biases = self._variable_on_cpu('biases', b_shape, tf.constant_initializer(0.0))
            pre_activation = tf.nn.bias_add(conv, biases)
            conv_result = tf.nn.relu(pre_activation, name=scope.name)  # relu激活
            self._activation_summary(conv_result)
            return conv_result

    def fully_collection_layer(self,name,input,w_shape,initializer,b_shape,keep_prob=None,wd=None):
        with tf.variable_scope(name) as scope:
            kernel = self._variable_with_weight_decay('weights', shape=w_shape, initializer=initializer)  # 权值矩阵
            biases = self._variable_on_cpu('biases', b_shape, tf.constant_initializer(0.0))
            relu_result = tf.nn.relu_layer(input,kernel,biases)  # relu激活
            drop_result = tf.nn.dropout(relu_result,keep_prob=keep_prob,name=scope.name)
            self._activation_summary(drop_result)
            return drop_result

    def inference(self, images):
        """向前传播
        """
        # 第一层卷积
        conv1_1 = self.conv2d_layer('conv1',images,[3,3,3,64],initializer_conv2d,[64])
        conv1_2 = self.conv2d_layer('conv2',conv1_1,[3,3,64,64],initializer_conv2d,[64])
        # pool1 最大池化
        pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
        # norm1 增加一个LRN处理,可以增强模型的泛化能力
        #norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

        # 第二层卷积
        #conv2_1 = self.conv2d_layer('conv2_1', norm1, [3, 3, 32, 64], 0.1, [64])
        conv2_1 = self.conv2d_layer('conv2_1', pool1, [3, 3, 64, 128], initializer_conv2d, [128])
        conv2_2 = self.conv2d_layer('conv2_2', conv2_1, [3, 3, 128, 128], initializer_conv2d, [128])
        # 这次先进行LRN处理
        #norm2 = tf.nn.lrn(conv2_2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
        # 最大池化
        #pool2 = tf.nn.max_pool(norm2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
        pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

        # 第三层卷积
        conv3_1 = self.conv2d_layer('conv3_1', pool2, [3, 3, 128, 256], initializer_conv2d, [256])
        conv3_2 = self.conv2d_layer('conv3_2', conv3_1, [3, 3, 256, 256], initializer_conv2d, [256])
        conv3_3 = self.conv2d_layer('conv3_3', conv3_2, [3, 3, 256, 256], initializer_conv2d, [256])
        conv3_4 = self.conv2d_layer('conv3_4', conv3_3, [3, 3, 256, 256], initializer_conv2d, [256])
        # 这次先进行LRN处理
        #norm3 = tf.nn.lrn(conv3_3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        # 最大池化
        #pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        pool3 = tf.nn.max_pool(conv3_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        # 第四层卷积
        conv4_1 = self.conv2d_layer('conv4_1', pool3, [3, 3, 256, 512], initializer_conv2d, [512])
        conv4_2 = self.conv2d_layer('conv4_2', conv4_1, [3, 3, 512, 512], initializer_conv2d, [512])
        conv4_3 = self.conv2d_layer('conv4_3', conv4_2, [3, 3, 512, 512], initializer_conv2d, [512])
        conv4_4 = self.conv2d_layer('conv4_4', conv4_3, [3, 3, 512, 512], initializer_conv2d, [512])
        # 这次先进行LRN处理
        # norm3 = tf.nn.lrn(conv3_3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        # 最大池化
        # pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        pool4 = tf.nn.max_pool(conv4_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        # 第五层卷积
        conv5_1 = self.conv2d_layer('conv5_1', pool4, [3, 3, 512, 512], initializer_conv2d, [512])
        conv5_2 = self.conv2d_layer('conv5_2', conv5_1, [3, 3, 512, 512], initializer_conv2d, [512])
        conv5_3 = self.conv2d_layer('conv5_3', conv5_2, [3, 3, 512, 512], initializer_conv2d, [512])
        conv5_4 = self.conv2d_layer('conv5_4', conv5_3, [3, 3, 512, 512], initializer_conv2d, [512])
        # 这次先进行LRN处理
        # norm3 = tf.nn.lrn(conv3_3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')
        # 最大池化
        # pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
        pool5 = tf.nn.max_pool(conv5_4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        reshape = tf.reshape(pool5, [self.batch_size, -1])
        dim = reshape.get_shape()[1].value
        fully_collection1 = self.fully_collection_layer('fully_collection1',reshape,[dim, 4096],initializer_fully,[4096],0.5, wd=0.004)

        fully_collection2 = self.fully_collection_layer('fully_collection2',fully_collection1,[4096,4096],initializer_fully,[4096],0.5,wd=0.004)

        with tf.variable_scope('softmax_linear') as scope:
            weights = self._variable_with_weight_decay('weights', [4096, self.num_classes], initializer_fully, wd=0.0)
            biases = self._variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
            softmax_linear = tf.add(tf.matmul(fully_collection2, weights), biases, name=scope.name)

            self._activation_summary(softmax_linear)

        return softmax_linear


    def loss(self, logits, labels):
        """损失函数
        """
        # Calculate the average cross entropy loss across the batch.
        labels = tf.cast(labels, tf.int64)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits, name='cross_entropy_per_example')
        cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
        tf.add_to_collection('losses', cross_entropy_mean)

        # The total loss is defined as the cross entropy loss plus all of the weight
        # decay terms (L2 loss).
        return tf.add_n(tf.get_collection('losses'), name='total_loss')

    def evaluation(self, logits, labels, k=1):
        """评估函数
        :param logits: 预测
        :param labels: 标签
        """
        correct = tf.nn.in_top_k(logits, labels, k=k)
        # correct = tf.equal(self.predictions, tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    def _add_loss_summaries(self, total_loss):
        """增加损失摘要
        """
        # Compute the moving average of all individual losses and the total loss.
        loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
        losses = tf.get_collection('losses')
        loss_averages_op = loss_averages.apply(losses + [total_loss])

        # Attach a scalar summary to all individual losses and the total loss; do the
        # same for the averaged version of the losses.
        for l in losses + [total_loss]:
            # Name each loss as '(raw)' and name the moving average version of the loss
            # as the original loss name.
            tf.summary.scalar(l.op.name + ' (raw)', l)
            tf.summary.scalar(l.op.name, loss_averages.average(l))

        return loss_averages_op

    def train_op(self, total_loss, global_step):
        """训练操作
        """
        num_batches_per_epoch = self.num_train_examples / self.batch_size   # 每轮的批次数
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)   # 多少步衰减

        # 基于步数，以指数方式衰减学习率。
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, decay_steps, LEARNING_RATE_DECAY_FACTOR, staircase=True)
        tf.summary.scalar('learning_rate', lr)
        # 损失移动平均
        loss_averages_op = self._add_loss_summaries(total_loss)

        with tf.control_dependencies([loss_averages_op]):
            opt = tf.train.AdamOptimizer(lr)  # 优化器
            grads = opt.compute_gradients(total_loss)    # 梯度

        # 应用梯度
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)   # 训练操作

        # 为可训练的变量添加直方图
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)

        # 为梯度添加直方图
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)

        # 跟踪所有可训练变量的移动平均线
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
            train_op = tf.no_op(name='train')

        return train_op

    def train_step(self, sess, summary_writer):
        """单步训练
        """
        _, step, cur_loss, cur_acc = sess.run([self.train_op, self.global_step, self.loss, self.accuracy])
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, cur_loss, cur_acc))
        # 存储摘要
        if step % 100 == 0:
            summary_str = sess.run(self.summary)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

    def train(self, filename,is_shuffle, out_dir):
        """训练
        """
        with tf.Graph().as_default():
            sess = tf.Session(config=self.session_conf)
            with sess.as_default():

                self.global_step = tf.contrib.framework.get_or_create_global_step()

                with tf.device('/cpu:0'):
                    images, labels = TFRecord.createBatch(filename, self.batch_size,is_shuffle)
                logits = self.inference(images)
                self.loss = self.loss(logits, labels)
                self.train_op = self.train_op(self.loss, self.global_step)
                self.accuracy = self.evaluation(logits, labels)
                self.summary = tf.summary.merge_all()

                # 保存点设置
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")  # 模型存储前缀
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
                summary_writer = tf.summary.FileWriter(out_dir + "/summary", sess.graph)

                # 初始化所有变量
                ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
                if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    sess.run(tf.global_variables_initializer())

                tf.train.start_queue_runners(sess=sess)

                for step in range(self.max_steps):
                    self.train_step(sess, summary_writer)  # 训练
                    cur_step = tf.train.global_step(sess, self.global_step)
                    # checkpoint_every 次迭代之后 保存模型
                    if cur_step % self.checkpoint_every == 0 and cur_step != 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=cur_step)
                        print("Saved model checkpoint to {}\n".format(path))