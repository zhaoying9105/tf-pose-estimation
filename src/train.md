## 设置 日志 logger

```
logger = logging.getLogger('train')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
```

## 设置参数
```
parser = argparse.ArgumentParser(description='Training codes for Openpose using Tensorflow')
parser.add_argument('--model', default='mobilenet', help='model name')
parser.add_argument('--datapath', type=str, default='/root/coco/annotations')
parser.add_argument('--imgpath', type=str, default='/root/coco/')
parser.add_argument('--batchsize', type=int, default=96)
parser.add_argument('--gpus', type=int, default=1)
parser.add_argument('--max-epoch', type=int, default=30)
parser.add_argument('--lr', type=str, default='0.01')
parser.add_argument('--modelpath', type=str, default='/data/private/tf-openpose-models-2018-1/')
parser.add_argument('--logpath', type=str, default='/data/private/tf-openpose-log-2018-1/')
parser.add_argument('--checkpoint', type=str, default='')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--remote-data', type=str, default='', help='eg. tcp://0.0.0.0:1027')

parser.add_argument('--input-width', type=int, default=368)
parser.add_argument('--input-height', type=int, default=368)
args = parser.parse_args()
```

## 网络入口

```
input_node = tf.placeholder(tf.float32, shape=(args.batchsize, args.input_height, args.input_width, 3), name='image')
vectmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 38), name='vectmap')
heatmap_node = tf.placeholder(tf.float32, shape=(args.batchsize, output_h, output_w, 19), name='heatmap')
```

这三个应该会 连接在一起

## 不太确定


![](http://img.blog.csdn.net/20180226154156753?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvSGliZXJjcmFmdA==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)
这个应该是单个网络
```
    for gpu_id in range(args.gpus):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                net, pretrain_path, last_layer = get_network(args.model, q_inp_split[gpu_id])
                vect, heat = net.loss_last()
                output_vectmap.append(vect)
                output_heatmap.append(heat)
                outputs.append(net.get_output())
```

这个是多个网络
```
                l1s, l2s = net.loss_l1_l2()
                for idx, (l1, l2) in enumerate(zip(l1s, l2s)):
                    loss_l1 = tf.nn.l2_loss(tf.concat(l1, axis=0) - q_vect_split[gpu_id], name='loss_l1_stage%d_tower%d' % (idx, gpu_id))
                    loss_l2 = tf.nn.l2_loss(tf.concat(l2, axis=0) - q_heat_split[gpu_id], name='loss_l2_stage%d_tower%d' % (idx, gpu_id))
                    losses.append(tf.reduce_mean([loss_l1, loss_l2]))

                last_losses_l1.append(loss_l1)
                last_losses_l2.append(loss_l2)

    outputs = tf.concat(outputs, axis=0)

```

## 优化

注意loss的计算


```
    with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
        # define loss
        total_loss = tf.reduce_sum(losses) / args.batchsize
        total_loss_ll_paf = tf.reduce_sum(last_losses_l1) / args.batchsize
        total_loss_ll_heat = tf.reduce_sum(last_losses_l2) / args.batchsize
        total_loss_ll = tf.reduce_mean([total_loss_ll_paf, total_loss_ll_heat])

        # define optimizer
        step_per_epoch = 121745 // args.batchsize
        global_step = tf.Variable(0, trainable=False)
        if ',' not in args.lr:
            starter_learning_rate = float(args.lr)
            learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                       decay_steps=10000, decay_rate=0.33, staircase=True)
        else:
            lrs = [float(x) for x in args.lr.split(',')]
            boundaries = [step_per_epoch * 5 * i for i, _ in range(len(lrs)) if i > 0]
            learning_rate = tf.train.piecewise_constant(global_step, boundaries, lrs)

```


## summary 不用管

```
 tf.summary.scalar("loss", total_loss)
    tf.summary.scalar("loss_lastlayer", total_loss_ll)
    tf.summary.scalar("loss_lastlayer_paf", total_loss_ll_paf)
    tf.summary.scalar("loss_lastlayer_heat", total_loss_ll_heat)
    tf.summary.scalar("queue_size", enqueuer.size())
```


## valid loss 是什么？

```
    valid_loss = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_paf = tf.placeholder(tf.float32, shape=[])
    valid_loss_ll_heat = tf.placeholder(tf.float32, shape=[])
    sample_train = tf.placeholder(tf.float32, shape=(4, 640, 640, 3))
    sample_valid = tf.placeholder(tf.float32, shape=(12, 640, 640, 3))
    train_img = tf.summary.image('training sample', sample_train, 4)
    valid_img = tf.summary.image('validation sample', sample_valid, 12)
    valid_loss_t = tf.summary.scalar("loss_valid", valid_loss)
    valid_loss_ll_t = tf.summary.scalar("loss_valid_lastlayer", valid_loss_ll)
```
