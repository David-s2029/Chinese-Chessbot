import pyrealsense2 as rs
import numpy as np
import cv2
import datetime

# 初始化RealSense管道
pipeline = rs.pipeline()

# 创建一个配置对象
config = rs.config()

# 告诉管道我们希望使用相机默认的RGB相机流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# 开始流
profile = pipeline.start(config)

try:
    while True:
        # 等待获取一组帧（颜色）
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        # 如果没有获取到帧，就继续循环
        if not color_frame:
            continue

        # 将图像数据转换为numpy数组
        color_image = np.asanyarray(color_frame.get_data())

        # 显示图像
        cv2.imshow('RealSense', color_image)

        key = cv2.waitKey(1)
        # 按 's' 键保存图片
        if key == ord('s'):
            img_name = f"RS_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            cv2.imwrite(img_name, color_image)
            print(f"Image {img_name} saved.")
        # 按 'q' 键退出
        elif key == ord('q'):
            break

finally:
    # 停止流
    pipeline.stop()

cv2.destroyAllWindows()
