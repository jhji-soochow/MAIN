# DoGAIN调试记录

## 1. DoG部分调试 

训练数据选用50张，加速迭代，目的是找到最适合的DoG分解方式。

### 2020年07月07日10:37:51
考虑如下参数

    x1 = self.G(x, torch.tensor([[[[1/2]]]]))
    x2 = self.G(x1, torch.tensor([[[[1]]]]))
    x3 = self.G(x2, torch.tensor([[[[2]]]]))
    x4 = self.G(x3, torch.tensor([[[[4]]]]))

输出三个尺度的结果

    return x-x2, x2-x4, x4
    

### 2020年07月07日11:30:31
补充了一个AIN的baseline，同样是在50张数据上训练的，这个AIN是没有DoG的，用于比较


### 2020年07月07日14:41:44
加速迭代   将数据量下调至20， patchsize 下调至48，这样可以多跑几个模型
关停上面两个模型

重新再新配置上继续

### 2020年07月07日17:25:52
增加两个尺度的对比实验，主要想看看尺度数量的影响

### 2020年07月07日22:40:07
在dogain2中，增加了归一化操作， 即对输入的每个实例都减均值，并除以方差，在重建阶段做响应的逆操作

### 2020年07月08日11:32:48
对于高斯核大小为$5$，尺度值为$8$的情况下，理论上：
高频的均值为$0$，标准差为 $255-0.0413*255\approx244.5$
低频部分均值为$127.5$，标准差为$127.5$

这个命名为DoGAIN3
    h_mean, h_std = 0, 244.5
    x_h = (x_h - h_mean) / h_std # 这一步操作会使得原来的数更小
    x_h = self.AIN_1(x_h)
    x_h = x_h * h_std + h_mean 

    l_mean, l_std = 127.5, 127.5
    x_l = (x_l - l_mean) / l_std
    x_l = self.AIN_3(x_l)
    x_l = x_l * l_std + l_mean


### 2020年07月08日14:36:27
对比一下 这个命名为DoGAIN4
    h_mean, h_std = 0, 255
    x_h = (x_h - h_mean) / h_std
    x_h = self.AIN_1(x_h)
    x_h = x_h * h_std + h_mean

    l_mean, l_std = 127.5, 127.5
    x_l = (x_l - l_mean) / l_std
    x_l = self.AIN_3(x_l)
    x_l = x_l * l_std + l_mean


### 2020年07月08日15:04:31
DoGAIN3 and 4 对两个层都做了归一化，但这样的结果比不上标准化，DoGAIN5中，将h层的标准差改为 127.5，与l层一致


### 
研究一下BN





