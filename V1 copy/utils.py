import numpy as np

class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        x_before, x_after, y = sample['x_before'], sample['x_after'], sample['y']


        h, w = x_before.shape[1:] #grabbing h and w assuming [C, H, W]
        new_h, new_w = self.output_size

        assert h >= new_h , "Output height is larger than original height"
        assert w >= new_w, "Output width is larger than original width"

        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)

        x_before = x_before[:, top:top+new_h, left:left+new_w]
        x_after = x_after[:, top:top+new_h, left:left+new_w]
        print("Y shape, dataloader:", y.shape)
        y = y[:, top:top+new_h, left:left+new_w]

        return {'x_before': x_before, 'x_after': x_after, 'y': y}
    