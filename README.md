# MAIN-PyTorch

![](/figs/network.png)

This repository is an PyTorch implementation of the paper **"Multi-scale Attention-aware Inception Network for Image Interpolation"** from **Tranactions on Image Processing (TIP)**.

**Note:** This is a re-implemented version for readers to train and test our code. For only conducting a comparison with our results in the MAIN paper.  You can download some results from [here](https://drive.google.com/drive/u/0/folders/1yO2dyG3sbnCAbSF787Os-6TtlMlc2jzj), including the *ground-truth images*, our *produced results*, and the *exploited evluation methods* (in Matlab).  

In this repository, you can train your own model from scratch, and test the trained on the provided ground-truth images. Or, you can download our pre-trained model and use it to enlarge your images.

The Dependencies are listed as follows:
* Python 3.6
* PyTorch = 1.7
* numpy
* skimage
* imageio
* matplotlib
* tqdm 
* visdom (optional)

To use our code, please clone this repository into your device and then change the directory to ``MAIN`` by using the following two commands:
```bash
git clone https://github.com/thstkdgus35/EDSR-PyTorch
cd EDSR-PyTorch
```
### How to train MAIN
The [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset that contains 800 training images is exploited to train our model. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar).

Unpack the tar file to "/dataset". We recommend you to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

If you have enough RAM (>= 32GB), you can use ``--ext bin`` argument to pack all DIV2K images in one binary file.

You can train the proposed *attention-aware inception network* (AIN) by yourself. 

```bash
cd src       # You are now in */EDSR-PyTorch/src
./train_MAIN.sh
```

2. You can use "--visdom" command to monitor your training process. To enable it, you should install the "Visdom" package first. See [here](https://github.com/fossasia/visdom) for some tutorials.  


## Quick start (Demo)

1. 

You can test our super-resolution algorithm with your own images. Place your images in ``test`` folder. (like ``test/<your_image>``) We support **png** and **jpeg** files.

Run the script in ``src`` folder. Before you run the demo, please uncomment the appropriate line in ```demo.sh``` that you want to execute.
```bash
cd src       # You are now in */EDSR-PyTorch/src
sh demo.sh
```



## Citation
If you find our work useful in your research or publication, please cite our work:

[1] J. Ji, B. Zhong and K. -K. Ma, "Image Interpolation Using Multi-Scale Attention-Aware Inception Network," in IEEE Transactions on Image Processing, vol. 29, pp. 9413-9428, 2020, doi: 10.1109/TIP.2020.3026632.

@ARTICLE{9210165,
  author={J. {Ji} and B. {Zhong} and K. -K. {Ma}},
  journal={IEEE Transactions on Image Processing}, 
  title={Image Interpolation Using Multi-Scale Attention-Aware Inception Network}, 
  year={2020},
  volume={29},
  number={},
  pages={9413-9428},
  doi={10.1109/TIP.2020.3026632}}


