import random

from imgaug import augmenters as iaa
import math
import cv2
import numpy as np
import pywt
# from data_process.load_single_data import RESIZE_SIZE

RESIZE_SIZE = 112

def random_cropping(image, target_shape=(32, 32, 3), is_random = True):
    if image.shape[0] == RESIZE_SIZE:
        resize_size = RESIZE_SIZE
    else:
        resize_size = int(RESIZE_SIZE / 2)
    resize_size = RESIZE_SIZE
    image = cv2.resize(image, (resize_size, resize_size))
    target_h, target_w,_ = target_shape
    
    # print(target_w,target_h)
    height, width, _ = image.shape
    # print(height, width)
    # print(target_h, target_w,height, width)

    if is_random:
        start_x = random.randint(0, width - target_w)
        start_y = random.randint(0, height - target_h)
    else:
        start_x = ( width - target_w ) // 2
        start_y = ( height - target_h ) // 2

    zeros = image[start_y:start_y+target_h,start_x:start_x+target_w,:]
    return zeros

def TTA_1_cropps_color(image, target_shape=(48, 48, 3)):
    # image = cv2.resize(image, (target_shape[0], target_shape[1]))
    image = random_cropping(image, target_shape, is_random=True)
    image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
    return image


def TTA_3_cropps_color(image, target_shape=(48, 48, 3)):
    if image.shape[0] == RESIZE_SIZE:
        resize_size = RESIZE_SIZE
    else:
        resize_size = int(RESIZE_SIZE / 2)
    image = cv2.resize(image, (resize_size, resize_size))
    width, height, _ = image.shape
    target_w, target_h, _ = target_shape
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    # start_x = 40
    # start_y = 40

    starts = [[start_x - target_w, start_y - target_w],[start_x - target_w, start_y],[start_x - target_w, start_y + target_w],
              [start_x, start_y - target_w],[start_x, start_y],[start_x, start_y + target_w],
              [start_x + target_w, start_y - target_w],[start_x + target_w, start_y],[start_x + target_w, start_y + target_w],
              ]
    images = []

    index = random.sample(range(1, 9), 3)
    count = 0

    for start_index in starts:
        count = count + 1
        image_ = image.copy()
        x, y = start_index
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + target_w >= resize_size:
            x = resize_size - target_w - 1
        if y + target_h >= resize_size:
            y = resize_size - target_h - 1
        for i in index:
            if i == count:
                zeros = image_[x:x + target_w, y: y + target_h]
                image_ = zeros.copy()
                images.append(image_.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
    images = np.concatenate(images, axis=0)
    return images


def TTA_4_cropps(image, target_shape=(32, 32, 3)):
    target_w, target_h = target_shape[0], target_shape[1]
    image = cv2.resize(image, (target_w, target_h))
    images = []
    image_ = image.copy()
    zeros = image_

    zeros = np.fliplr(zeros)
    image_flip_lr = zeros.copy()

    zeros = np.flipud(zeros)
    image_flip_lr_up = zeros.copy()

    zeros = np.fliplr(zeros)
    image_flip_up = zeros.copy()
    # print(image.shape)
    images.append(image_.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
    images.append(image_flip_lr.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
    images.append(image_flip_up.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
    images.append(image_flip_lr_up.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
    images = np.concatenate(images, axis=0)
    return images


def TTA_9_cropps_color(image, target_shape=(48, 48, 3)):
    if image.shape[0] == RESIZE_SIZE:
        resize_size = RESIZE_SIZE
    else:
        resize_size = int(RESIZE_SIZE / 2)
    image = cv2.resize(image, (resize_size, resize_size))
    width, height, _ = image.shape
    target_w, target_h, _ = target_shape
    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2
    # start_x = 40
    # start_y = 40

    starts = [[start_x - target_w, start_y - target_w],[start_x - target_w, start_y],[start_x - target_w, start_y + target_w],
              [start_x, start_y - target_w],[start_x, start_y],[start_x, start_y + target_w],
              [start_x + target_w, start_y - target_w],[start_x + target_w, start_y],[start_x + target_w, start_y + target_w],
              ]
    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index
        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x + target_w >= resize_size:
            x = resize_size - target_w - 1
        if y + target_h >= resize_size:
            y = resize_size - target_h - 1
        zeros = image_[x:x + target_w, y: y + target_h]
        image_ = zeros.copy()
        images.append(image_.reshape([1, target_shape[0], target_shape[1], target_shape[2]]))
    images = np.concatenate(images, axis=0)
    return images


def TTA_18_cropps(image, target_shape=(32, 32, 3)):
    if image.shape[0] == RESIZE_SIZE:
        resize_size = RESIZE_SIZE
    else:
        resize_size = int(RESIZE_SIZE / 2)
    image = cv2.resize(image, (resize_size, resize_size))

    width, height, d = image.shape
    target_w, target_h, d = target_shape

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],
              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],
              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],]

    images = []
    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= resize_size:
            x = resize_size - target_w-1
        if y + target_h >= resize_size:
            y = resize_size - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()
        zeros = np.fliplr(zeros)
        image_flip = zeros.copy()

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
    images = np.concatenate(images, axis=0)
    return images


def TTA_27_cropps(image, target_shape=(32, 32, 3)):
    # print(image.shape)
    if image.shape[0] == RESIZE_SIZE:
        resize_size = RESIZE_SIZE
    else:
        resize_size = int(RESIZE_SIZE / 2)
    image = cv2.resize(image, (resize_size, resize_size))

    width, height, d = image.shape
    target_w, target_h, d = target_shape
    # print(width, height)
    # print(target_shape)

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= resize_size:
            x = resize_size - target_w-1
        if y + target_h >= resize_size:
            y = resize_size - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        # zeros = np.fliplr(zeros)
        # image_flip_up = zeros.copy()
        # print(image.shape)
        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        # images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
    images = np.concatenate(images, axis=0)
    return images


def TTA_36_cropps(image, target_shape=(32, 32, 3)):
    # print(image.shape)
    if image.shape[0] == RESIZE_SIZE:
        resize_size = RESIZE_SIZE
    else:
        resize_size = int(RESIZE_SIZE / 2)
    resize_size = RESIZE_SIZE
    image = cv2.resize(image, (resize_size, resize_size))

    width, height, d = image.shape
    target_w, target_h, d = target_shape
    # print(width, height)
    # print(target_shape)

    start_x = ( width - target_w) // 2
    start_y = ( height - target_h) // 2

    starts = [[start_x, start_y],

              [start_x - target_w, start_y],
              [start_x, start_y - target_w],
              [start_x + target_w, start_y],
              [start_x, start_y + target_w],

              [start_x + target_w, start_y + target_w],
              [start_x - target_w, start_y - target_w],
              [start_x - target_w, start_y + target_w],
              [start_x + target_w, start_y - target_w],
              ]

    images = []

    for start_index in starts:
        image_ = image.copy()
        x, y = start_index

        if x < 0:
            x = 0
        if y < 0:
            y = 0

        if x + target_w >= resize_size:
            x = resize_size - target_w-1
        if y + target_h >= resize_size:
            y = resize_size - target_h-1

        zeros = image_[x:x + target_w, y: y+target_h, :]

        image_ = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_lr = zeros.copy()

        zeros = np.flipud(zeros)
        image_flip_lr_up = zeros.copy()

        zeros = np.fliplr(zeros)
        image_flip_up = zeros.copy()

        # print(image.shape)

        images.append(image_.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
        images.append(image_flip_lr_up.reshape([1,target_shape[0],target_shape[1],target_shape[2]]))
    images = np.concatenate(images, axis=0)
    return images

def random_resize(img, probability = 0.5,  minRatio = 0.2):
    if random.uniform(0, 1) > probability:
        return img

    ratio = random.uniform(minRatio, 1.0)

    h = img.shape[0]
    w = img.shape[1]

    new_h = int(h*ratio)
    new_w = int(w*ratio)

    img = cv2.resize(img, (new_w,new_h))
    img = cv2.resize(img, (w, h))
    return img


def wave_t(image):
    img = image
    # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    b, g, r = cv2.split(img)
    cAr, (cHr, cVr, cDr) = pywt.dwt2(r, 'haar')
    cAg, (cHg, cVg, cDg) = pywt.dwt2(g, 'haar')
    cAb, (cHb, cVb, cDb) = pywt.dwt2(b, 'haar')
    image_wave = np.stack([cHr, cVr, cDr, cHg, cVg, cDg, cHb, cVb, cDb], axis=-1)

    return image_wave

def color_augumentor(image, label=None, target_shape=(32, 32, 3), is_infer=False, isLocal=False):
    if is_infer:
        # augment_img = iaa.Sequential([iaa.Fliplr(0),])

        augment_img = iaa.Sequential([iaa.Fliplr(0.5),
                                      # iaa.Add(value=(-10, 10), per_channel=True),  # protocol 3&4 wo
                                      # iaa.GammaContrast(gamma=(0.9, 1.1)),  # protocol 3&4 wo
                                      ])
        image = augment_img.augment_image(image)
        if isLocal and target_shape[0] < RESIZE_SIZE - 10:
            image = TTA_36_cropps(image, target_shape)
            # image = TTA_1_cropps_color(image, target_shape)
        else:
            # image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
            image = TTA_4_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([iaa.Fliplr(0.5),iaa.Flipud(0.5),iaa.Affine(rotate=(-30, 30)),], random_order=True)
        image = augment_img.augment_image(image)
        if isLocal:
            image = random_resize(image)
            # if image.shape[2] == target_shape[0]:
            image = random_cropping(image, target_shape, is_random=True)
        return image


def strong_color_augumentor(image, label=None, target_shape=(32, 32, 3), is_infer=False, isLocal=False):
    """Augmentor with stronger perturbations to improve generalization to unseen attack types.
    Simulates display artifacts (blur, noise, color shift, compression) that occur in replay attacks."""
    if is_infer:
        augment_img = iaa.Sequential([iaa.Fliplr(0.5)])
        image = augment_img.augment_image(image)
        if isLocal and target_shape[0] < RESIZE_SIZE - 10:
            image = TTA_36_cropps(image, target_shape)
        else:
            image = TTA_4_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Affine(rotate=(-30, 30)),
            # Display artifact simulation
            iaa.Sometimes(0.3, iaa.GaussianBlur(sigma=(0.5, 2.0))),
            iaa.Sometimes(0.3, iaa.AdditiveGaussianNoise(scale=(5, 25))),
            iaa.Sometimes(0.3, iaa.Add(value=(-25, 25), per_channel=True)),
            iaa.Sometimes(0.3, iaa.GammaContrast(gamma=(0.6, 1.4))),
            iaa.Sometimes(0.2, iaa.JpegCompression(compression=(40, 85))),
            # Domain-invariant augmentations
            iaa.Sometimes(0.2, iaa.Grayscale(alpha=(0.5, 1.0))),
            iaa.Sometimes(0.15, iaa.ChannelShuffle(p=1.0)),
            # CutOut regularization
            iaa.Sometimes(0.3, iaa.CoarseDropout(p=(0.02, 0.1), size_percent=(0.02, 0.25))),
        ], random_order=True)
        image = augment_img.augment_image(image)
        if isLocal:
            image = random_resize(image)
            image = random_cropping(image, target_shape, is_random=True)
        return image


def depth_augumentor(image, label=None, target_shape=(32, 32, 3), is_infer=False, isLocal=False):

    if is_infer:
        augment_img = iaa.Sequential([iaa.Fliplr(0),])
        image =  augment_img.augment_image(image)
        if isLocal:
            image = TTA_36_cropps(image, target_shape)
            # image = TTA_9_cropps_color(image, target_shape)
            # image = TTA_1_cropps_color(image, target_shape)
        else:
            # image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
            image = TTA_4_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(-30, 30)),], random_order=True)
        image = augment_img.augment_image(image)
        if isLocal:
            image = random_resize(image)
            image = random_cropping(image, target_shape, is_random=True)
        return image


def ir_augumentor(image, label=None, target_shape=(32, 32, 3), is_infer=False, isLocal=False):
    if is_infer:
        augment_img = iaa.Sequential([iaa.Fliplr(0),])
        image =  augment_img.augment_image(image)
        if isLocal:
            image = TTA_36_cropps(image, target_shape)
        else:
            # image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
            image = TTA_4_cropps(image, target_shape)
        return image

    else:
        augment_img = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(-30, 30)),], random_order=True)
        image = augment_img.augment_image(image)
        if isLocal:
            image = random_resize(image)
            image = random_cropping(image, target_shape, is_random=True)
        return image


def new_augumentor(image, label=None, target_shape=(32, 32, 3), is_infer=False, isLocal=False):
    if is_infer:
        augment_img = iaa.Sequential([iaa.Fliplr(0),])
        image =  augment_img.augment_image(image)
        if isLocal:
            image = TTA_36_cropps(image, target_shape)
        else:
            image = image.reshape([1, image.shape[0], image.shape[1], image.shape[2]])
        return image

    else:
        augment_img = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5), iaa.Affine(rotate=(-30, 30)),], random_order=True)
        image = augment_img.augment_image(image)
        if isLocal:
            image = random_resize(image)
            image = TTA_3_cropps_color(image, target_shape)
            # image = random_cropping(image, target_shape, is_random=True)
        return image


def augumentor_train(color, label, target_shape=(32, 32, 3)):
    augment_img_neg = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Add(value=(0,20),per_channel=True),
        iaa.Add(value=(-30 ,30) ,per_channel=True),  # protocol 3 +-10; 4 +- 30;
        iaa.GammaContrast(gamma=(0.5 ,1.5)),  # protocol 3/4 1+-0.5
        # iaa.Affine(rotate=(-30, 30)),
    ])

    augment_img_pos = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Add(value=(-10,10),per_channel=True),
        # iaa.Add(value=(0,10),per_channel=True), #protocol 3&4 wo
        # iaa.GammaContrast(gamma=(0.9, 1.1)), #protocol 3&4 wo
    ])

    if random.random() < 0.5:
        if int(label) > 0:
            color = augment_img_pos.augment_image(color)
        else:
            color = augment_img_neg.augment_image(color)

    # if random.random() < 0.1:   # protocol 3
    #    color = CutOut(color)      # protocol 3

    # color = (color - 127.5) / 128
    # color = RandomErasing(color)  # protocol 3&4 wo
    # color = TTA_9_cropps_color(color, target_shape)
    color = random_cropping(color, target_shape, is_random=True)

    return color


## augment for validation and test
def augumentor_test(color, label=None, target_shape=(32, 32, 3)):
    augment_img = iaa.Sequential([
        iaa.Fliplr(0.5),
        # iaa.Add(value=(-10,10),per_channel=True), #protocol 3&4 wo
        # iaa.GammaContrast(gamma=(0.9, 1.1)), #protocol 3&4 wo
    ])
    color = augment_img.augment_image(color)
    color = (color - 127.5) / 128
    # color = TTA_9_cropps_color(color, target_shape)
    color = TTA_36_cropps(color, target_shape)

    return color

def augumentor_OULU(color, label=None, target_shape=(32, 32, 3), is_infer = False, isLocal=False):
    if is_infer:
        augment_img = iaa.Sequential([
            iaa.Fliplr(0.5),
        ])
        color = augment_img.augment_image(color)
        color = TTA_36_cropps(color, target_shape)

        return color
    else:
        augment_img_neg = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Add(value=(-30, 30), per_channel=True),
            iaa.GammaContrast(gamma=(0.5, 1.5)),
        ])

        augment_img_pos = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Add(value=(-10,10),per_channel=True),
        ])

        if random.random() < 0.5:
            if int(label) > 0:
                color = augment_img_pos.augment_image(color)
            else:
                color = augment_img_neg.augment_image(color)

        color = random_cropping(color, target_shape, is_random=True)

        return color