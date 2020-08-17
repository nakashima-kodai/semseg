import os


def is_image_file(filename):
    img_extensions = [ '.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG' ]
    return any(filename.endswith(extension) for extension in img_extensions)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images
