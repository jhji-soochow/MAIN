# MAIN-PyTorch

![](/figs/network.png)

This repository is a PyTorch implementation of the paper **"Multi-scale Attention-aware Inception Network for Image Interpolation"** from **Transactions on Image Processing (TIP)**.

**Note:** This is a re-implemented version for readers to train and test our code. For only conducting a comparison with our results in the MAIN paper.  You can download some results from [here](https://drive.google.com/drive/u/0/folders/1yO2dyG3sbnCAbSF787Os-6TtlMlc2jzj), including the *ground-truth images*, our *produced results*, and the *exploited evluation methods* (in Matlab).  

In this repository, you can train your own model from scratch, and test the trained model on the provided ground-truth images. 
Or, you can download our pre-trained model and use it to interpolate your images. [coming soon]

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
git clone https://github.com/jhji-soochow/MAIN.git
cd MAIN
```
### How to train?

The [DIV2K](http://www.vision.ee.ethz.ch/%7Etimofter/publications/Agustsson-CVPRW-2017.pdf) dataset that contains 800 training images is exploited to train our model. Please download it from [here](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar). Then, unpack the tar file to "/dataset". 

Some of our codes are borrowed form the of the repository of [EDSR](https://github.com/thstkdgus35/EDSR-PyTorch). According to EDSR, it is recommend to pre-process the images before training. This step will decode all **png** files and save them as binaries. Use ``--ext sep_reset`` argument on your first run. You can skip the decoding part and use saved binaries with ``--ext sep`` argument.

For the test datasets, download the ``GT_images`` from [here](https://drive.google.com/drive/u/0/folders/1yO2dyG3sbnCAbSF787Os-6TtlMlc2jzj) and unpack them to "/dataset/myBenchmark". Your benchmark dataset should be:
```
cd dataset/myBenchmark
tree -L 1
.
├── BSDS100
├── Manga109
├── Set15
├── Set18
├── Set5
├── Urban100
└── Urban12
```

Once the datasets are ready, you can train our ***attention-aware inception network*** (AIN) by yourself. For your convenience, we provide a training script for network training. You can run the script by: 

```bash
cd src       # You are now in */MAIN/src
# 1. Before you run the demo, please uncomment the appropriate line in that you 
#    want to execute.
# 2. If you have more than one GPU cards, you can choose to use a specific GPU 
#    by setting the device number with ``CUDA_VISIBLE_DEVICES=x``.
./train_MAIN.sh
```

We also provide the command of "--visdom" to monitor your model's performance on the benchmark datasets during network training. To enable it, you should install the "Visdom" package first. See [here](https://github.com/fossasia/visdom) for some tutorials.  


TODO: 
We will upload some pretrained models and inference code of MAIN.

<!-- #### How to interpolate your test images with our published pretrained models.

You can test our MAIN method with your own images. Place your images into any place ( our default folder is ``test``). 

Then run the script of ``run_MAINnet.sh`` in ``src``: 
```bash
cd src
sh train_MAIN.sh
``` -->

#### Citation
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


