from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math

import numpy as np
import tensorflow as tf

import CNN
import TFRecord

tf.flags.DEFINE_string("filename", "data/test.TFRecords", "测试数据所在文件")
tf.flags.DEFINE_string("checkpoint_dir", "runs/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_integer("batch_size", 40, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_classes", 2, "类别 (default: 2)")
tf.flags.DEFINE_integer("num_examples", 800, "训练样本 (default: 64)")
tf.flags.DEFINE_boolean('is_shuffle', False, "Is shuffle")

# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

cnns = CNN.CNNClassify(FLAGS.batch_size,FLAGS.num_classes,FLAGS.num_examples)

def evaluate():
    with tf.Graph().as_default() as g:
        with tf.Session() as sess:
            
            images, labels = TFRecord.createBatch(FLAGS.filename, FLAGS.batch_size, FLAGS.is_shuffle)
            
            logits = cnns.inference(images)

            # Calculate predictions.
            top_k_op = tf.nn.in_top_k(logits, labels, 1)

            saver = tf.train.Saver(tf.trainable_variables())
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)

            coord = tf.train.Coordinator()
            threads = []
            try:
                for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                    threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

                num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
                true_count = 0  # 计算正确的数量
                total_sample_count = num_iter * FLAGS.batch_size  # 总数量
                step = 0
                while step < num_iter and not coord.should_stop():
                    predictions = sess.run([top_k_op])
                    # print(np.sum(predictions))
                    true_count += np.sum(predictions)
                    step += 1

                # Compute precision @ 1.
                precision = true_count / total_sample_count
                print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            except Exception as e:  # pylint: disable=broad-except
                coord.request_stop(e)

            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=10)


evaluate()