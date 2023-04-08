import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def show_ims(ims, columns=5, figsize=(20, 10), title=None, fontsize=None):
    """
    Show the given PIL images using matplotlib.

    Args:
        ims: A list of the PIL images
        columns: The number of columns
        figsize: The size of the matplotlib figure
        title: A list with the same length as `imgs`, giving the
            title to display above each respective image. If a single string is
            given, it is used as main title.
    """
    if title is not None and isinstance(title, list):
        assert len(title) == len(ims)

    fig = plt.figure(figsize=figsize)
    for i, im in enumerate(ims):
        plt.subplot(len(ims) // columns + 1, columns, i + 1)
        plt.imshow(np.asarray(im))
        plt.grid(b=False)
        plt.axis('off')

        if title is not None and isinstance(title, list):
            plt.title(title[i])
    if isinstance(title, str):
        fig.suptitle(title, fontsize=fontsize)


def show_imgs(imgs, columns=5, figsize=(20, 10), title=None):
    """
    Show the given images using matplotlib.

    Args:
        imgs: A list of the image paths
        columns: The number of columns
        figsize: The size of the matplotlib figure
        title: A list with the same length as `imgs`, giving the
            title to display above each respective image.
    """
    return show_ims([Image.open(img) for img in imgs],
                    columns=columns, figsize=figsize,
                    title=titles)


def scale_im(im, scale=0.3):
    """
    Return a rescaled version of the given PIL Image.
    """
    im = im.copy()
    im.thumbnail([int(s*scale) for s in im.size])
    return im


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def resize_height(im: Image, new_height: int):
    r = new_height / im.height
    if r != 1:
        new_width = int(im.width * r)
        im = im.resize((new_width, new_height))
    return im


def resize_width(im: Image, new_width: int):
    r = new_width / im.width
    if r != 1:
        new_height = int(im.height * r)
        im = im.resize((new_width, new_height))
    return im


def add_border(im, width=1, color='black'):
    draw = ImageDraw.Draw(im)
    draw.rectangle((0, 0, im.width - width, im.height - width),
                   outline=color, width=width)
    return im


def equal_height_grid(ims, row_height, ims_per_row, padding=5, border=False):
    ims = [resize_height(im, row_height) for im in ims]

    if border:
        ims = [add_border(im) for im in ims]

    ims = [
        np.pad(
            np.array(im),
            ((padding,padding), (padding, padding), (0, 0)),
            constant_values=255
        )
        for im in ims
    ]

    im_rows = [
        Image.fromarray(np.hstack(im_row))
        for im_row in chunks(ims, ims_per_row)
    ]

    new_width = im_rows[0].width

    im_rows = [
        resize_width(im, new_width)
        for im in im_rows
    ]

    return Image.fromarray(np.vstack(im_rows))