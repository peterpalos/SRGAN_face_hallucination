import utilities as ut
from os import listdir
from PIL import Image

image_name_list = listdir('original')

target_size = (256, 256)
input_size = (64, 64)
for ind, im in enumerate(image_name_list):
    with Image.open('original/' + im) as image:
        image = ut.convert_image_mode_if_not_rgb(image)
        image = ut.pad_image_if_not_square(image)
        image = ut.resize_to(image, target_size)
        image.save('train_hr/train_hr/' +  '{}.jpg'.format(ind), 'JPEG')
        image = ut.resize_to(image, input_size)
        image.save('train_lr/train_lr/' +  '{}.jpg'.format(ind), 'JPEG')