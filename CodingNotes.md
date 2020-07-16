## create_pretraining_data.py

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
以后要学会直接用tensorflow的参数体系了，而不是只有argparse。
实例化一个FLAGS--->用各种DEFINE_string/integer/bool/float方法定义参数--->写一个main()函数，里面可以使用这些参数---->最后在“__main__”里面定义flags.mark_flag_as_required  还有使用tf.app.run()
* 这些tensorflow  flags在使用的时候似乎就当正常的python变量使用即可
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

####  4 random包都被怎么使用的?
* 在flag的参数里面有一个random_seed，随机过程以后要是有意识控制的话，请在argparse或者tf.flags里增加random seed
* [random api doc](https://docs.python.org/3/library/random.html)
* random 产生的随机数是伪随机数
* 哪些场景用到了随机呢？
    * 文件名的shuffle
    * 随机确定sequence的长度。使用了一个random.random产生浮点数代表probability，又用了一个random.randint(最小值，最大值)生成想要的长度
    * 随机选择文本的index
```
import random
# 定义一个参数，产生种子
flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
# 使用随机种子得到一个随机数生成器
rng = random.Random(FLAGS.random_seed)
# 如何给list进行随机shuffle
rng.shuffle(all_documents)
rng.shuffle(instances)
# 如何产生一个随机的浮点数
rng.random()
# 产生一个随机的整数
rng.randint(最小值，最大值)

```

#### 5 tf.compat.v1.app.run
这是v1时候的机制了，v2的tensorflow就没再保留了
Runs the program with an optional 'main' function and 'argv' list.
```
tf.compat.v1.app.run(
    main=None, argv=None
)
```

#### 6 tokenization
* tokenization指的是splitting up a larger body of text into smaller lines, words or even creating words for a non-English language.
* 但是网上大多的教程都是nltk的，不过bert这个代码的tokenization并不是使用这个包，而是自己写了一个tokenization.py文件

#### 7 TFRecordWriter
* 本代码把writer写在一个list里面。而且存的格式就是tfrecord格式
```python
# 1. 构造
tf.python.io.TFRecordWriter(文件名)
```

#### 8. list有extend和pop方法
* append和extend的区别
```
x = [1, 2, 3]
x.append([4, 5]) # x = [1, 2, 3, [4, 5]]
x.extend([4, 5])  # x = [1, 2, 3, 4, 5]
```
```
tokens_b = []
tokens_b.extend(random_document[j])
len(tokens_b)
tokens_b.pop()
```

#### 9. tf.logging
```
tf.logging.set_verbosity(tf.logging.INFO)
tf.logging.info("*** Example ***")
tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))
tf.logging.info("Wrote %d total instances", total_written)
f.logging.info("  %s", output_file)
```
#### 10. tf.gfile.Glob(文件夹pattern)
就类似于glob


#### 11. 代码逻辑
##### 11.1
* 设置flags和各种参数---> main里面搞一个tokenizer，以后用于切割文本 ---->然后把文本的路径准备好--->切割成tokenizer之后write

## extract_features.py的代码学习
### 1 参数
flags    Define_bool/integer/float/string   mark_flags_as_required   tf.app.run()
### 2 logging
tf.logging.set_verbosity(tf.logging.INFO)      
### 3 codecs
* The codecs module provides stream and file interfaces for transcoding data in your program. It is most commonly used to work with Unicode text, but other encodings are also available for other purposes.
```
with codecs.getwriter("utf-8")(tf.gfile.Open(FLAGS.output_file,
                                               "w")) as writer:
      writer.write(json.dumps(output_json) + "\n")                                       
```
