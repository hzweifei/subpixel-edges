from matplotlib import pyplot as plt
import cv2
import numpy as np
from subpixel_edges import subpixel_edges
import time

if __name__=="__main__":
    image=cv2.imread("circle.jpg",cv2.IMREAD_GRAYSCALE)
    start=time.time()
    e = subpixel_edges(image, 50, 1, 2)
    end=time.time()
    print(f"检测所需时间为：{end-start} s")
    points=np.column_stack((e.x, e.y)).astype(np.float32)
    ellipse = cv2.fitEllipseAMS(points)
    # 输出拟合结果
    print(
        f"亚像素拟合的椭圆：中心({ellipse[0][0]:.3f}, {ellipse[0][1]:.3f}), 长轴半径 {ellipse[1][0] / 2:.3f}, "
        f"短轴半径 {ellipse[1][1] / 2:.3f}")
