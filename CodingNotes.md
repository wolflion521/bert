#### 1 OrderedDict
https://github.com/wolflion521/bert/blob/acc576f88345638d989de3a1054ba0136013ffa3/create_pretraining_data.py#L132
https://docs.python.org/zh-cn/3/library/collections.html            
OrderedDict和namedtuple学习一下使用。           
OrderedDict比Dict的映射性能低一些，它适合在需要记住插入顺序的场景中使用。           
这份代码里OrderedDict是在讲数据变成tf.train.Example实例的使用被用到。就当dict使用了，往key里面传value。原来tf.train.Features()构造函数里面可以传OrderedDict。      
#### 2 tf.train.Example的使用
##### 2.1
https://github.com/wolflion521/bert/blob/acc576f88345638d989de3a1054ba0136013ffa3/create_pretraining_data.py#L141
python的list--->经过tf.train.XXXList函数--->再经过tf.train.Feature函数--->变成了feature。    
众多feature通过OrderedDict组合到一起---->tf.train.Features变成features组---> tf.train.Example最终变成example。         
数据保存流程。tf.train.Example---> SerializeToString()--->tf.python_io.TFRecordWriter的write函数将其写进一个文件里            

```
import collections
def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature
def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature
features = collections.OrderedDict()
features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
features["next_sentence_labels"] = create_int_feature([next_sentence_label])
tf_example = tf.train.Example(features=tf.train.Features(feature=features))
```
##### 2.2 

#### 3 参数

```
import tensorflow as tf
flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")
DEFINE_string        DEFINE_bool           DEFINE_integer      DEFINE_float

def main(_):
    instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

if __name__ == "__main__":
  flags.mark_flag_as_required("input_file")
  flags.mark_flag_as_required("output_file")
  flags.mark_flag_as_required("vocab_file")
  tf.app.run()

```
