from PIL import Image, ImageOps

def convert_image_mode_if_not_rgb(image):
    if image.mode != 'RGB':
        print('Image mode not RGB, converted')
        return image.convert(mode='RGB')
    print('Image mode RGB')
    return image

def pad_image_if_not_square(image):
    if image.height != image.width:
        new_size = max(image.height, image.width)
        delta_w = new_size - image.width
        delta_h = new_size - image.height
        padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
        print('Image aspect ratio not 1:1, padded')
        return ImageOps.expand(image, padding)
    print('Image aspect ratio 1:1')
    return image  

def resize_to(image, new_size=(64, 64)):
    if image.size != new_size:
        print('Image resized to {}'.format(new_size))
        return image.resize(new_size)
    print('Image already the correct size')
    return image
