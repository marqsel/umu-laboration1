import numpy as np
from PIL import Image
import PIL.ImageOps


# Iterates over all combinations and yields them
def iter_combinations(*iterables):
    if not len(iterables):
        yield tuple()
    else:
        for item in iterables[0]:
            for group in iter_combinations(*iterables[1:]):
                yield (item,) + group


# Transform a comma line argument and returns a list. If item type is specified, each value is transformed to that type.
def cmd_arg_to_list(value, item_type=None, seperator=','):
    values = value.split(seperator)
    if item_type:
        values = map(item_type, values)
    return list(values)


# Normalizes and transformes the input value used for the network.
def to_grey(X):
    X = X.astype(np.float32) / 255.0
    X = X.reshape(*(X.shape + (1,)))
    return X


# Inverts an image's colors.
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


# Transforms an image from IO and returns input used when predicting a number.
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
