- [human](#human)
- [BodyPart](#bodypart)
  - [属性](#%E5%B1%9E%E6%80%A7)
  - [get_part_name](#getpartname)
- [PoseEstimator](#poseestimator)
  - [属性](#%E5%B1%9E%E6%80%A7)
  - [non_max_suppression](#nonmaxsuppression)
  - [estimate](#estimate)
  - [score_pairs](#scorepairs)
  - [get_score](#getscore)
- [TfPoseEstimator](#tfposeestimator)
  - [init](#init)
      - [不知道在做什么](#%E4%B8%8D%E7%9F%A5%E9%81%93%E5%9C%A8%E5%81%9A%E4%BB%80%E4%B9%88)
  - [\_quantize\_img](#quantizeimg)
  - [draw_humans](#drawhumans)
  - [\_get\_scaled\_img](#getscaledimg)
      - [get_base_scale](#getbasescale)
  - [\_crop\_roi](#croproi)
  - [inference](#inference)


# human

# BodyPart

## 属性

- BodyPart：身体部位的索引，比如 nose 是 0

- x, y: 身体部位的坐标

- score：身体部的置信度

## get_part_name

返回身体部位的名字


# PoseEstimator

## 属性

heatmap_supress，heatmap_gaussian 两种热力图的模式
NMS_Threshold：NMS 是non_max_suppression 的缩写

## non_max_suppression

NMS_Threshold
非最大值抑制


## estimate

根据得到的heat_mat, paf_mat 进行估计

```python
    nms = PoseEstimator.non_max_suppression(plain, 5, _NMS_Threshold)
    coords.append(np.where(nms >= _NMS_Threshold))
```
给定阈值进行过滤

看不懂，不知道 CocoPairs, CocoPairsNetwork  是什么意思

```python
        # score pairs
        pairs_by_conn = list()
        for (part_idx1, part_idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
            pairs = PoseEstimator.score_pairs(
                part_idx1, part_idx2,
                coords[part_idx1], coords[part_idx2],
                paf_mat[paf_x_idx], paf_mat[paf_y_idx],
                heatmap=heat_mat,
                rescale=(1.0 / heat_mat.shape[2], 1.0 / heat_mat.shape[1])
            )

            pairs_by_conn.extend(pairs)
```


## score_pairs
也看不懂

## get_score

# TfPoseEstimator

## init

先加载计算图



#### 不知道在做什么

```python
        # warm-up
        self.persistent_sess.run(
            self.tensor_output,
            feed_dict={
                self.tensor_image: [np.ndarray(shape=(target_size[1], target_size[0], 3), dtype=np.float32)]
            }
        )
```

## \_quantize\_img

把图片的值从（0-255）缩放到（0-1）

## draw_humans

- npimg 是原始图片，作为背景
- human 是用于画关键点的信息

用opencv画人形图，包括关键点和连线


## \_get\_scaled\_img

图片缩放
#### get_base_scale

自己看代码
```
        get_base_scale = lambda s, w, h: max(self.target_size[0] / float(w), self.target_size[1] / float(h)) * s
```

四种情况

- scale is None 
     直接 resize
- isinstance(scale, float):
  - ratio_x,ratio_y 可以看做resize 之后比 target_size 多出来的比例
  - resize之后的大图片取右下角和 target_size 同样大小的图片
- 
  


## \_crop\_roi


cropped = npimg[y:y+target_h, x:x+target_w]
resize之后的大图片取右下角和 target_size 同样大小的图片

## inference

推断
```python
  heatMats = output[:, :, :, :19]
  pafMats = output[:, :, :, 19:]
```

对 多个 heatMat, pafMat 进行处理
```python

 for heatMat, pafMat, info in zip(heatMats, pafMats, infos):
            w, h = int(info[2]*mat_w), int(info[3]*mat_h)
            heatMat = cv2.resize(heatMat, (w, h))
            pafMat = cv2.resize(pafMat, (w, h))
            x, y = int(info[0] * mat_w), int(info[1] * mat_h)

            if TfPoseEstimator.ENSEMBLE == 'average':
                # average
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] += heatMat[max(0, -y):, max(0, -x):, :]
                resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                resized_cntMat[max(0,y):y+h, max(0, x):x+w, :] += 1
            else:
                # add up
                resized_heatMat[max(0, y):y + h, max(0, x):x + w, :] = np.maximum(resized_heatMat[max(0, y):y + h, max(0, x):x + w, :], heatMat[max(0, -y):, max(0, -x):, :])
                resized_pafMat[max(0,y):y+h, max(0, x):x+w, :] += pafMat[max(0, -y):, max(0, -x):, :]
                resized_cntMat[max(0, y):y + h, max(0, x):x + w, :] += 1

        if TfPoseEstimator.ENSEMBLE == 'average':
            self.heatMat = resized_heatMat / resized_cntMat
            self.pafMat = resized_pafMat / resized_cntMat
        else:
            self.heatMat = resized_heatMat
            self.pafMat = resized_pafMat / (np.log(resized_cntMat) + 1)
```