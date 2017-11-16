from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import CNN


tf.flags.DEFINE_string("filename", "data/train.TFRecords", "训练数据所在文件")
tf.flags.DEFINE_string("out_dir", "./runs", "模型输出路径")
tf.flags.DEFINE_integer("num_classes", 2, "类别 (default: 2)")
tf.flags.DEFINE_integer("batch_size", 40, "批次大小 (default: 64)")
tf.flags.DEFINE_integer("num_train_examples", 24200, "训练样本 (default: 64)")
tf.flags.DEFINE_boolean('is_shuffle', True, "Is shuffle")

# 打印参数
FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

cnns = CNN.CNNClassify(FLAGS.batch_size,FLAGS.num_classes,FLAGS.num_train_examples)

cnns.train(FLAGS.filename, FLAGS.is_shuffle, FLAGS.out_dir)