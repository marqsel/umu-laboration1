import numpy as np
from PIL import Image
import PIL.ImageOps


def iter_combinations(*iterables):
    if not len(iterables):
        yield tuple()
    else:
        for item in iterables[0]:
            for group in iter_combinations(*iterables[1:]):
                yield (item,) + group


def to_grey(X):
    X = X.astype(np.float32) / 255.0
    X = X.reshape(*(X.shape + (1,)))
    return X


def invert_image(image):
    if image.mode == 'RGBA':
        r,g,b,a = image.split()
        rgb_image = Image.merge('RGB', (r,g,b))

        inverted_image = PIL.ImageOps.invert(rgb_image)

        r2,g2,b2 = inverted_image.split()

        image = Image.merge('RGBA', (r2,g2,b2,a))

    else:
        image = PIL.ImageOps.invert(image)

    return image


def transform_image(image, width, height, invert=False, filename=None):
    img = Image.open(image)

    if invert:
        img = invert_image(img)
    img = img.resize((width, height))

    img = img.convert('L')

    if filename:
        img.save(filename, 'PNG')

    img = np.asarray(img)

    img = to_grey(img)

    return img
