import numpy as np

def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.741, 0.534,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


COLORS_CLAZZ = np.insert(colormap(), obj=3, values=100 / 255, axis=1)
BORDER = 10
COLORS_OK = np.array(((255, 0, 0, 100), (0, 255, 0, 100))) / 255

def blend_img(background, label_mask, gamma=2.2):
    overlay_rgba = COLORS_CLAZZ[label_mask]
    alpha = overlay_rgba[:, :, 3]
    over_corr = np.float_power(overlay_rgba[:, :, :3], gamma)
    bg_corr = np.float_power(background, gamma)
    return np.float_power(over_corr * alpha[..., None] + (1 - alpha)[..., None] * bg_corr, 1 / gamma)  # dark magic
    # partially taken from https://en.wikipedia.org/wiki/Alpha_compositing#Composing_alpha_blending_with_gamma_correction


def create_vis(rgb, label, prediction):
    if rgb.shape[0] == 3:
        rgb = rgb.transpose(1, 2, 0)
    if len(prediction.shape) == 3:
        prediction = np.argmax(prediction, 0)

    h, w, _ = rgb.shape

    gt_map = blend_img(rgb, COLORS_CLAZZ[label])  # we can index colors, wohoo!
    pred_map = blend_img(rgb, COLORS_CLAZZ[prediction])
    ok_map = blend_img(rgb, COLORS_OK[
        (label == prediction).astype('u1')])  # but we cannot do it by boolean, otherwise it won't work
    canvas = np.ones((h * 2 + BORDER, w * 2 + BORDER, 3))
    canvas[:h, :w] = rgb
    canvas[:h, -w:] = gt_map
    canvas[-h:, :w] = pred_map
    canvas[-h:, -w:] = ok_map

    canvas = (np.clip(canvas, 0, 1) * 255).astype('u1')
    return Image.fromarray(canvas)
