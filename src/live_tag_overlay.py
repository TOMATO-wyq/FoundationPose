"""Tag换算箱体框；RGB颜色青色，假设35cm箱体、Tag贴面正中心。"""
import itertools
import cv2
import numpy as np


def draw_tag_cube(rgb, pose, k):
    corners=np.array(list(itertools.product((-.175,.175),repeat=3)))
    cam=corners@pose[:3,:3].T+pose[:3,3]
    if np.any(cam[:,2]<=0):return rgb
    pixels=cam@k.T
    pixels=np.clip(np.rint(pixels[:,:2]/pixels[:,2:]),-10000,10000).astype(int)
    for i in range(8):
        for j in range(i+1,8):
            if np.count_nonzero(corners[i]!=corners[j])==1:
                cv2.line(rgb,tuple(pixels[i]),tuple(pixels[j]),(0,255,255),2,cv2.LINE_AA)
    return rgb
