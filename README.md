# Green Planet Project
#### SmartLab - Deep Learning in practice - home project
Upscaling low resolution images of people's faces to higher resolutions by using Single Image Super-Resolution Deep Neural Networks.

co-author: [Márton Hévizi](https://github.com/habarcs)

# SRGAN implementation
The input of the network will be low resolution portraits. The goal is to upscale the pictures and accurately fill in the missing information, predicting what the individual may have looked like.
Implementation of [this article](https://arxiv.org/pdf/1609.04802.pdf).
Based on [this code](https://github.com/eriklindernoren/Keras-GAN/blob/master/srgan/srgan.py).

### Database
For training our network, we use multiple databases with over 100000 pictures.
Links to the databases:
- [Matting Human Datasets](https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets)
- [IDOC Mugshots](https://www.kaggle.com/elliotp/idoc-mugshots)

All these pictures are different sizes and we still need inputs and targets so we use our image preprocessor script.
For this repository we only included a small number of pictures as an example.

### Data preparing
Our project aims to upscale 64x64 images to 256X256, a scaling factor of 4x. To train our network, we use the 256X256 pictures as targets and their downscaled version as inputs.

The image preprocessor script creates our data. It takes the original pictures from the database one by one and creates the input and target pictures.

First it changes the image mode to RGB then pads the images with a black border to make the aspect ratio 1:1. Now it can be scaled down to 64X64 and 256X256 to create our inputs and targets. Now every input can be represented as a 64X64X3 multidimensional array, where each value is between 0-255. We didn't normalize them, because keras can do that for us.

Input and target images are saved with the same name but in different directories for easy identification and pairing.
