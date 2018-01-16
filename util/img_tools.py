# -*- coding:utf-8 -*- 
# 图片处理工具
# Author：WangQi、tudoudou

import os
import random
import numpy as np
import math
from PIL import Image, ImageFont, ImageDraw


def img_uniform_scale(img_or_path, save_path=None, height_keep=50, width_keep=None):
    """ 图片缩放

    Args:
        img_or_path: PIL对象或者输入图片路径
        save_path: 输出图片路径
        height_keep: 保持高度为`height_keep` px，此时`width_keep`为`None`，则宽度保持等比缩放
        width_keep: 保持宽度为`width_keep` px，此时`height_keep`为`None`，则高度保持等比缩放
    Returns:
        如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。
    """

    if height_keep is None and width_keep is None:
        raise ValueError('height_keep and width_keep can not be `None` at the same time.')
    if (height_keep is not None and height_keep <= 0) or \
            (width_keep is not None and width_keep <= 0):
        raise ValueError('height_keep OR width_keep must be greater than `0`.')

    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    img_size = img.size

    if height_keep is not None and width_keep is None:
        new_size = (int((float(img_size[0]) / img_size[1]) * height_keep), height_keep)
    elif width_keep is not None and height_keep is None:
        new_size = (width_keep, int((float(img_size[1]) / img_size[0]) * width_keep))
    else:
        new_size = (width_keep, height_keep)

    out = img.resize(new_size, Image.ANTIALIAS)
    if save_path:
        out.save(save_path)
    else:
        return out


def rm_white_bg(img_or_path, convert_pix_post=195, save_path=None):
    """ 移除黑白色的jpg图片中的白色背景，即将白色转换为透明色。返回一张png图片。

    Args:
        img_or_path: PIL对象或者输入图片路径
        convert_pix_post: 白色值转透明的像素下限，即大于等于195像素值就转为透明
        save_path: 修改后的图片保存路径

    Returns:
        如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。
    """

    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    png_img = img.convert('RGBA')
    png_arr = np.array(png_img)
    max_val_rgb = np.max(png_arr[:, :, :3], 2)
    tmp = np.maximum(max_val_rgb, convert_pix_post)
    tmp = np.equal(tmp, convert_pix_post)  # 等于convert_pix_post意味着RGBA中A为255
    tmp = tmp.astype(np.uint8) * 255
    png_arr[:, :, 3] = tmp
    if save_path is not None:
        Image.fromarray(png_arr).save(save_path)
    else:
        return Image.fromarray(png_arr)


def png_paste(upper_pic, bg_pic, save_path=None, horizontal='center', vertical='center'):
    """合并图片
    Args:
        upper_pic: 上图层图片路径或者PIL对象
        bg_pic: 背景图片路径或者PIL对象
        save_path: 合成图片路径
        horizontal: 水平位置选择，默认居中，可用参数：left(左) center(居中) right(右) random(随机)
        vertical: 垂直位置选择，默认居中，可用参数：top(顶部) center(居中) bottom(底部) random(随机)

    Returns:
        如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。

        示例位置：\n
         ver\\\hor \t left \t align \t right \n
        top \n
        align \n
        bottom
    """

    if isinstance(upper_pic, str):
        upper_pic = Image.open(upper_pic)
    if isinstance(bg_pic, str):
        bg_pic = Image.open(bg_pic)
    if bg_pic.size[0] < upper_pic.size[0] or bg_pic.size[1] < upper_pic.size[1]:
        raise ValueError('bg_pic(background picture) must bigger of upper_pic(front picture).')
    if horizontal not in ['left', 'center', 'right', 'random']:
        raise ValueError('horizontal must in left, center, right or random.')
    if vertical not in ['top', 'center', 'bottom', 'random']:
        raise ValueError('vertical must in top, center, bottom or random.')

    if horizontal == 'left':
        x = 0
    elif horizontal == 'center':
        x = (bg_pic.size[0] - upper_pic.size[0]) / 2
    elif horizontal == 'right':
        x = bg_pic.size[0] - upper_pic.size[0]
    else:
        x = (bg_pic.size[0] - upper_pic.size[0]) * random.random()
    if vertical == 'top':
        y = 0
    elif vertical == 'center':
        y = (bg_pic.size[1] - upper_pic.size[1]) / 2
    elif vertical == 'bottom':
        y = bg_pic.size[1] - upper_pic.size[1]
    else:
        y = (bg_pic.size[1] - upper_pic.size[1]) * random.random()

    try:
        bg_pic.paste(upper_pic, (int(x), int(y)), mask=upper_pic)
    except:
        print('upper_pic(front picture)\'s type is not RGBA.')
        bg_pic.paste(upper_pic, (int(x), int(y)))

    if save_path:
        bg_pic.save(save_path)
    else:
        return bg_pic


def clip(img_or_path, save_path=None):
    """裁剪图片主要内容

    Args:
        img_or_path: 图片或者PIL对象
        save_path: 保存路径

    Returns:
        如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。
    """

    if isinstance(img_or_path, str):
        img = Image.open(img_or_path)
    else:
        img = img_or_path

    img = rm_white_bg(img)
    a = np.array(img.split()[3])  # 提取 Alpha 通道

    if a.shape[0] <= 0 and a.shape[1] <= 0:
        raise ValueError('This img\'s must bigger 0')

    def _clip(list):
        """返回列表第一个正数

        Args:
            list: 待判断列表

        Returns:
            列表第一个正数,若没有,返回0
        """
        index = 0
        for i in list:
            if i > 0:
                return index
            index += 1
        return 0

    x = a.sum(axis=0)
    y = a.sum(axis=1)
    x1 = _clip(x)
    x_ = x[::-1]
    x2 = len(x) - _clip(x_)
    y1 = _clip(y)
    y_ = y[::-1]
    y2 = len(y) - _clip(y_)

    img = img.crop((x1, y1, x2, y2))  # 裁剪图片

    if save_path:
        return img.save(save_path)
    else:
        return img


def font2img(font, char, save_path=None, font_size=50, img_size=None, font_pos='middle', font_color=None):
    """利用字体库生成包含指定字符的png图片或者PIL对象

    Args:
        font: 字体文件路径
        char: 需要生成图片的一个字符
        save_path: 生成图片存储路径
        font_size: 字体大小
        font_pos: 文字生成位置，默认为`middle`中部，还可设置为`random`随机
        font_color: 字体颜色，默认为随机

    Returns:
        如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。
    """

    if font_pos not in ['middle', 'random']:
        raise ValueError('font_pos is must in middle or random')
    if not font_color:
        font_color = random.randint(0, 256) + random.randint(0, 256) * 255 + random.randint(0, 256) * 255 * 255
    if len(char) != 1:
        raise ValueError('char is not longer than two chars')

    font = ImageFont.truetype(font, font_size)
    m, n = font.getsize(char)

    if img_size is None:
        img_size = [m, n]
    im = Image.new("RGB", img_size, (255, 255, 255))
    dr = ImageDraw.Draw(im)

    if m > img_size[0] or n > img_size[1]:
        raise OverflowError('This font\'s size is out img_size')

    if font_pos == 'middle':
        x, y = (img_size[0] - m) / 2, (img_size[1] - n) / 2
    else:
        x, y = random.random() * (img_size[0] - m), random.random() * (img_size[1] - n)

    dr.text((x, y), char, font=font, fill=font_color)

    if save_path:
        im.save(save_path + char + '.png')
    else:
        return im


def img_concat(imgs, direction='horizontal', fixed_edge=None, align='middle', img_margin=None, save_path=None):
    ''' 横向或者纵向拼接imgs

    Args:
        imgs: 多个png图片列表，每张图片都是PIL对象
        direction: 纵向`vertical` 或者 横向`horizontal`拼接图片
        fixed_edge: 固定一条边进行缩放，当缩放高时，分别为`min_height`或者`max_height`或者为数值;
            当缩放宽时，分别为`min_height`或者`max_height`或者为数值;
            当无需缩放时，可以为`None`
        align: 图片对齐方式，包括居中`middle`，居左`left`，居右`right`，居上`top`，居下`bottom`，
            当`direction='horizontal'`时，可以填写居中`middle`，居上`top`，居下`bottom`，
            当`direction='vertical'`时，可以填写居中`middle`，居左`left`，居右`right`。
        img_margin: 拼接图片之间的间隔，可以为`None`或者一个数值，或者一个范围。指定范围是会随机取范围中的一个值。
        save_path: 拼接后的图片存储路径

    Returns:
        如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。
    '''
    for i in range(len(imgs)):
        assert imgs[i].mode == 'RGBA'

    if direction == 'horizontal':
        if fixed_edge == 'min_width' or fixed_edge == 'max_width':
            raise ValueError('when direction="horizontal", doesn\'t fixed width')
        if align == 'left' or align == 'right':
            raise ValueError('when direction="horizontal", doesn\'t set align="left" OR align="right"')
    else:
        if fixed_edge == 'min_height' or fixed_edge == 'max_height':
            raise ValueError('when direction="vertical", doesn\'t fixed height')
        if align == 'top' or align == 'bottom':
            raise ValueError('when direction="vertical", doesn\'t set align="top" OR align="bottom"')

    if isinstance(fixed_edge, int):
        if fixed_edge <= 0:
            raise ValueError('height_keep OR width_keep must be greater than `0`.')

    def get_min_max_edge(imgs, mode='height'):
        if mode == 'height':
            idx = 1
        else:
            idx = 0
        min = 99999999999
        max = 0
        for i in range(len(imgs)):
            if imgs[i].size[idx] > max:
                max = imgs[i].size[idx]
            if imgs[i].size[idx] < min:
                min = imgs[i].size[idx]
        return min, max

    def zero_pad_img(img, mode, width, height):
        if width:
            width = width - img.size[0]
        if height:
            height = height - img.size[1]
        img = np.array(img)
        left = right = top = bottom = 0
        if mode == 'middle':
            if width is None:
                top = math.ceil(height / 2.)
                bottom = height - top
            else:
                left = math.ceil(width / 2.)
                right = width - left
        elif mode == 'left':
            right = width
        elif mode == 'right':
            left = width
        elif mode == 'top':
            bottom = height
        else:
            top = height
        img = np.pad(img, [[top, bottom], [left, right], [0, 0]], 'constant')
        return Image.fromarray(img)

    def concat(imgs, direction='horizontal', img_margin=None):
        imgs = [np.array(img) for img in imgs]
        if direction == 'horizontal':
            if img_margin:
                if isinstance(img_margin, int):
                    margin = img_margin
                else:
                    if (isinstance(img_margin, tuple) or isinstance(img_margin, list)) \
                            and len(img_margin) == 2:
                        margin = random.randint(img_margin[0], img_margin[1])
                    else:
                        ValueError('`img_margin` type must be one of None、int、tuple、list')
                num_imgs = len(imgs)
                imgs_ = [np.pad(img, [[0, 0], [0, margin], [0, 0]], 'constant') \
                         for idx, img in enumerate(imgs) if idx < (num_imgs - 1)]
                imgs_.append(imgs[-1])
            res = np.concatenate(imgs_, 1)
        else:
            if img_margin:
                if isinstance(img_margin, int):
                    margin = img_margin
                else:
                    if (isinstance(img_margin, tuple) or isinstance(img_margin, list)) \
                            and len(img_margin) == 2:
                        margin = random.randint(img_margin[0], img_margin[1])
                    else:
                        ValueError('`img_margin` type must be one of None、int、tuple、list')
                num_imgs = len(imgs)
                imgs_ = [np.pad(img, [[0, margin], [0, 0], [0, 0]], 'constant') \
                         for idx, img in enumerate(imgs) if idx < (num_imgs - 1)]
                imgs_.append(imgs[-1])
            res = np.concatenate(imgs_, 0)
        return Image.fromarray(res)

    if fixed_edge == 'max_width':
        # 将所有图片的宽固定为图片中的最大宽度
        _, max_width = get_min_max_edge(imgs, 'width')
        imgs = [img_uniform_scale(img, None, None, max_width) for img in imgs]
    elif fixed_edge == 'min_width':
        # 将所有图片的宽固定为图片中的最小宽度
        min_width, _ = get_min_max_edge(imgs, 'width')
        imgs = [img_uniform_scale(img, None, None, min_width) for img in imgs]
    elif fixed_edge == 'max_height':
        # 将所有图片的高固定为图片中的最大高度
        _, max_height = get_min_max_edge(imgs, 'height')
        imgs = [img_uniform_scale(img, None, max_height, None) for img in imgs]
    elif fixed_edge == 'min_width':
        # 将所有图片的高固定为图片中的最小高度
        min_height, _ = get_min_max_edge(imgs, 'width')
        imgs = [img_uniform_scale(img, None, min_height, None) for img in imgs]

    if direction == 'horizontal':
        if isinstance(fixed_edge, int):
            imgs = [img_uniform_scale(img, None, fixed_edge, None) for img in imgs]
        if fixed_edge is None:
            _, max_height = get_min_max_edge(imgs, 'height')
            imgs = [zero_pad_img(img, align, None, max_height) for img in imgs]
    else:
        if isinstance(fixed_edge, int):
            imgs = [img_uniform_scale(img, None, None, fixed_edge) for img in imgs]
        if fixed_edge is None:
            _, max_width = get_min_max_edge(imgs, 'width')
            imgs = [zero_pad_img(img, align, max_width, None) for img in imgs]
    res_img = concat(imgs, direction, img_margin)
    if save_path:
        res_img.save(save_path)
    else:
        return res_img


class ImgRepo(object):
    '''从图片文件夹中生成指定大小的图片'''
    def __init__(self, img_dir):
        '''根据图片文件夹`img_dir`中的文件随机生成指定大小的图片
        当图片文件夹中的图片比生成的目标图片小时，可自动补0
        
        Args:
            img_dir: 背景图片文件夹
        '''
        self.all_path = []
        self.img_idx = 0
        for name in os.listdir(img_dir):
            path = os.path.join(img_dir, name)
            if os.path.isfile(path):
                if os.path.splitext(path)[1] == '.jpg' or os.path.splitext(path)[1] == '.png':
                    self.all_path.append(path)
        print('There are %d pictures in total.' % len(self.all_path))

    def get_one(self, size=[336, 224], save_path=None):
        '''生成一张指定大小的背景图片，返回PIL.Image对象或者一张图片。
        
        Args: 
            size: 生成的图片大小
            save_path: 输出图片路径
        Returns:
            如果`save_path == None`,则返回处理完成的Image对象，否则在指定路径保存。
        '''
        while True:
            try:
                if self.img_idx > len(self.all_path):
                    self.img_idx = 0
                img = Image.open(self.all_path[self.img_idx])
            except OSError:
                
                self.img_idx += 1
                print('read image faile!')
            else:
                self.img_idx += 1
                break
        
        pad_width = pad_height = 0
        if img.size[0] < size[0]:
            pad_width = size[0] - img.size[0]
        if img.size[1] < size[1]:
            pad_height = size[1] - img.size[1]
        
        if pad_width != 0 or pad_height != 0:
            img = np.array(img)
            img = np.pad(img, [[0, pad_height], [0, pad_width], [0, 0]], mode='constant')
            img = Image.fromarray(img)

        wider = img.size[0] - size[0]
        higher = img.size[1] - size[1]

        left = random.randint(0, wider)
        right = left + size[0]
        upper = random.randint(0, higher)
        lower = upper + size[1]

        img = img.crop([left, upper, right, lower])

        return img


# if __name__ == '__main__':
#     pass
#     #
#     # test fn
#     #
#     # img_uniform_scale(
#     #     '../data/processed_data/HWDB1.1/images/train/1002-c/阿606666.png',
#     #     '/tmp/test.png',
#     #     140, 240)
#
#     im1 = Image.open('../data/processed_data/HWDB1.1/images/train/1002-c/阿606666.png')
#     im2 = Image.open('../data/processed_data/HWDB1.1/images/train/1002-c/矮606675.png')
#     img_concat([im1, im2], direction='vertical', align='middle', save_path='/tmp/test2.png', img_margin=[10, 300])
