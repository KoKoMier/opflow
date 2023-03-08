import cv2
import numpy as np

#cap = cv2.VideoCapture("1.mp4")
cap = cv2.VideoCapture(0)

# 这里使用MP4V编解码器，FPS为30帧，分辨率为640x480
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (640, 480))

# 定义角点检测的参数
feature_params = dict(
    maxCorners=100,  # 最多多少个角点
    qualityLevel=0.3,  # 品质因子，在角点检测中会使用到，品质因子越大，角点质量越高，那么过滤得到的角点就越少
    minDistance=7  # 用于NMS，将最有可能的角点周围某个范围内的角点全部抑制
)

# 定义 lucas kande算法的参数
lk_params = dict(
    winSize=(10, 10),  # 这个就是周围点临近点区域的范围
    maxLevel=2  # 最大的金字塔层数
)

# 拿到第一帧的图像
ret, prev_img = cap.read()
prev_img_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

# 先进行角点检测，得到关键点
prev_points = cv2.goodFeaturesToTrack(prev_img_gray, mask=None, **feature_params)

# 制作一个临时的画布，到时候可以将新的一些画的先再mask上画出来，再追加到原始图像上
mask_img = np.zeros_like(prev_img)

x_last = 0
y_last = 0
while True:

    ret, curr_img = cap.read()
    if curr_img is None:
        print("video is over...")
        break
    curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
    try:
        # 光流追踪下
        curr_points, status, err = cv2.calcOpticalFlowPyrLK(prev_img_gray,
                                                        curr_img_gray,
                                                        prev_points,
                                                        None,
                                                        **lk_params)
        print(curr_points.shape[0])
        # print(status.shape)  # 取值都是1/0, 1表示是可以追踪到的，0表示失去了追踪的。
        good_new = curr_points[status == 1]
        good_old = prev_points[status == 1]
        # 绘制图像
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            
            qmask_img = cv2.line(mask_img, pt1=(int(a), int(b)), pt2=(int(c), int(d)), color=(0, 255, 255), thickness=1)
            #mask_img = cv2.circle(mask_img, center=(int(a), int(b)), radius=2, color=(255, 0, 0), thickness=1)
        x_err = b - d
        y_err = a - c
        x = x_err + x_last
        y = y_err + y_last
        x_last = x
        y_last = y
        cv2.putText(curr_img, 'x: {}  y:{}'.format(int(x),int(y)), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)    
        # 将画布上的图像和原始图像叠加，并且展示
        img = cv2.add(curr_img, mask_img)
        img = cv2.resize(img,(640,480))

        # 将帧写入输出视频中
        out.write(img)
        cv2.imshow("desct", img)

        if cv2.waitKey(60) & 0xFF == ord('q'):
            break

        # 更新下原始图像，以及重新得到新的点
        prev_img_gray = curr_img_gray.copy()
        prev_points = good_new.reshape(-1, 1, 2)
        img = cv2.rectangle(img, (0, 0), (640, 480), (0, 0, 0), -1)
        if len(prev_points) < 10:
            # 当匹配的太少了，就重新获得当前图像的角点
            prev_points = cv2.goodFeaturesToTrack(curr_img_gray, mask=None, **feature_params)
            mask_img = np.zeros_like(prev_img)  # 重新换个画布
    except:
        print("err")
cv2.destroyAllWindows()
cap.release()

