from matplotlib import pyplot as plt
import cv2
import numpy as np
from subpixel_edges import subpixel_edges
import time

if __name__=="__main__":
    image=cv2.imread("circle.jpg",cv2.IMREAD_GRAYSCALE)
    start=time.time()
    e = subpixel_edges(image, 50, 1, 2,non_maximum=False)
    print(1/np.mean(np.abs(e.curv)))
    end=time.time()
    print(f"检测所需时间为：{end-start} s")
    # points=np.column_stack((e.x, e.y)).astype(np.float32)
    points=e.sub_position
    ellipse = cv2.fitEllipseAMS(points)
    # 输出拟合结果
    print(
        f"亚像素拟合的椭圆：中心({ellipse[0][0]:.3f}, {ellipse[0][1]:.3f}), 长轴半径 {ellipse[1][0] / 2:.3f}, "
        f"短轴半径 {ellipse[1][1] / 2:.3f}")
    plt.imshow(image)
    plt.quiver(e.x, e.y, e.nx, -e.ny, scale=40)
    plt.show()
    print(len(e.x))

