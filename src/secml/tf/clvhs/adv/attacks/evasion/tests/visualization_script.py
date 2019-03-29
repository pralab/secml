from __future__ import print_function

from secml.array import CArray
from secml.figure import CFigure
from secml.utils import fm

digits = CArray([4, 9, 7])
img_h = 28
img_w = 28
distance = 'l2'


attack_names = ['FastGradientMethod']


def _show_adv(x0, y0, xopt, y_pred):
    """
    Show the original and the modified sample
    :param x0: original image
    :param xopt: modified sample
    """
    added_noise = abs(xopt - x0)  # absolute value of noise image

    if distance == 'l1':
        print("Norm of input perturbation (l1): ",
              added_noise.ravel().norm(ord=1))
    else:
        print("Norm of input perturbation (l2): ",
              added_noise.ravel().norm())

    fig = CFigure(height=5.0, width=15.0)
    fig.subplot(1, 3, 1)
    fig.sp.title(digits[y0].item())
    fig.sp.imshow(x0.reshape((img_h, img_w)), cmap='gray', interpolation=None)
    fig.subplot(1, 3, 2)
    fig.sp.imshow(
        added_noise.reshape((img_h, img_w)), cmap='gray', interpolation=None)
    fig.subplot(1, 3, 3)
    fig.sp.title(digits[y_pred].item())
    fig.sp.imshow(xopt.reshape((img_h, img_w)), cmap='gray',
                  interpolation=None)
    fig.savefig(
        fm.join(fm.abspath(__file__), 'evasion_mnist.pdf'), file_format='pdf')
    fig.show()


info_dir = fm.join(fm.abspath(__file__), 'attacks_data')

for attack_name in attack_names:

    x0 = CArray.load(fm.join(info_dir, attack_name + '_x0'))
    y0 = CArray.load(fm.join(info_dir, attack_name + '_y0')).astype(int)
    xopt = CArray.load(fm.join(info_dir, attack_name + '_adv_x'))
    y_pred = CArray.load(fm.join(info_dir, attack_name + '_ypred')).astype(int)

    _show_adv(x0, y0, xopt, y_pred)
