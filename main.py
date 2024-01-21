# -*- coding: utf-8 -*-

import cv2
#import time
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from skimage.feature import local_binary_pattern
import rospy
from dobot.srv import SetPTPCmd, SetPTPJumpParams, SetEndEffectorSuctionCup, SetHOMECmd,SetPTPCmdRequest,SetEndEffectorSuctionCupRequest,SetHOMECmdRequest

pieces = []
base_x = -160
base_y = 220
increment = 40 
base_z = -51
jump_height = 40
      
# 棋子类
class ChessPiece:
    def __init__(self, chess_type, color, x, y):
        self.color = color
        self.x = x
        self.y = y
        self.chess_type = chess_type

    def move(self, case, start, end):
        start_x = self.x
        start_y = self.y
        end_x = end 
        if (self.chess_type != "将") & (self.chess_type != "帅"):
            if self.chess_type != "士":
                if (self.chess_type != "象") & (self.chess_type != "相"):
                    if self.chess_type != "马":
                        if self.chess_type != "车":
                            if self.chess_type != "炮":
                                if case == 1:
                                    end_x = start_x
                                    end_y = start_y + end
                                elif case == 2:
                                    end_x = start_x
                                    end_y = start_y - end
                                elif case == 3:
                                    end_y = start_y
                                self.x = end_x
                                self.y = end_y
                                return start_x, start_y, end_x, end_y
                            else:
                                if case == 1:
                                    end_x = start_x
                                    end_y = start_y + end
                                elif case == 2:
                                    end_x = start_x
                                    end_y = start_y - end
                                elif case == 3:
                                    end_y = start_y
                                self.x = end_x
                                self.y = end_y
                                return start_x, start_y, end_x, end_y
                        else:
                            if case == 1:
                                end_x = start_x
                                end_y = start_y + end
                            elif case == 2:
                                end_x = start_x
                                end_y = start_y - end
                            elif case == 3:
                                end_y = start_y
                            self.x = end_x
                            self.y = end_y
                            return start_x, start_y, end_x, end_y
                    else:
                        distance = abs(start - end)
                        if case == 1:
                            if distance == 1:
                                end_y = start_y + 2
                            elif distance == 2:
                                end_y = start_y + 1
                        elif case == 2:
                            if distance == 1:
                                end_y = start_y - 2
                            elif distance == 2:
                                end_y = start_y - 1
                        self.x = end_x
                        self.y = end_y
                        return start_x, start_y, end_x, end_y
                else:
                    distance = abs(start - end)
                    if case == 1:
                        end_y = start_y + distance
                    elif case == 2:
                        end_y = start_y - distance
                    self.x = end_x
                    self.y = end_y
                    return start_x, start_y, end_x, end_y
            else:
                if case == 1:
                    end_y = start_y + 1
                elif case == 2:
                    end_y = start_y - 1
                self.x = end_x
                self.y = end_y
                return start_x, start_y, end_x, end_y
        else:
            if case == 1:
                if end == 1:
                    end_y = start_y + end
                    end_x = start_x
                else:
                    end_y = start_y + 1
            elif case == 2:
                if end == 1:
                    end_y = start_y - end
                    end_x = start_x
                else:
                    end_y = start_y - 1
            elif case == 3:
                end_x = end
                end_y = start_y
            self.x = end_x
            self.y = end_y
            return start_x, start_y, end_x, end_y

def chess_move(move): # 解析自然语言的行动指令
    chess_type = move[0]
    print(move[1])
    start = int(move[1])
    print(start)
    end = int(move[3])
    if "进" in move:
        case = 1
    elif "退" in move:
        case = 2
    elif "平" in move:
        case = 3
    else:
        raise ValueError("无法识别的移动方式")
    return chess_type, case, start, end

def transform_pixel_to_digital_x(pixel_x): # 将像素x坐标转化为棋盘x坐标
    block_blurr = 40/2
    block_space = 70
    for i in range(9):
        if pixel_x > 600 - i * block_space - block_blurr:
            if pixel_x < 600 - i * block_space + block_blurr:
                return i + 1
            else:
                continue
        continue
    
def transform_pixel_to_digital_y(pixel_y): # 将像素y坐标转化为棋盘y坐标
    block_blurr = 40/2
    block_space = 68
    for i in range(9):
        if pixel_y > 380 - i * block_space - block_blurr:
            if pixel_y < 380 - i * block_space + block_blurr:
                return i + 1
            else:
                continue
        continue
        
    

def dobot_ptp_client(x, y, z, r):
    rospy.wait_for_service('/DobotServer/SetPTPCmd')
    rospy.wait_for_service('DobotServer/SetPTPJumpParams')
    try:
        set_jump_para_client = rospy.ServiceProxy('DobotServer/SetPTPJumpParams',SetPTPJumpParams)
        dobot_ptp = rospy.ServiceProxy('/DobotServer/SetPTPCmd', SetPTPCmd)
        set_jump_para_client(40,120,0) 
        # 创建PTP命令请求
        ptp_req = SetPTPCmdRequest()
        ptp_req.ptpMode = 0  # 设置PTP模式
        ptp_req.x = x  # 设置目标X坐标
        ptp_req.y = y  # 设置目标Y坐标
        ptp_req.z = z  # 设置目标Z坐标
        ptp_req.r = r  # 设置目标R坐标

        # 调用服务
        response = dobot_ptp(ptp_req)
        return response.result
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def control_dobot_suction_cup(enable, is_sucked):
    rospy.wait_for_service('/DobotServer/SetEndEffectorSuctionCup')
    try:
        suction_cup_service = rospy.ServiceProxy('/DobotServer/SetEndEffectorSuctionCup', SetEndEffectorSuctionCup)

        # 创建吸盘命令请求
        suction_req = SetEndEffectorSuctionCupRequest()
        suction_req.enableCtrl = enable  # 启用或禁用吸盘控制
        suction_req.suck = is_sucked  # 吸附或释放

        # 调用服务
        response = suction_cup_service(suction_req)
        return response.result
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

def dobot_home():
    rospy.wait_for_service('/DobotServer/SetHOMECmd')
    try:
        home_service = rospy.ServiceProxy('/DobotServer/SetHOMECmd', SetHOMECmd)

        # 创建回零命令请求
        home_req = SetHOMECmdRequest()

        # 调用服务
        response = home_service(home_req)
        return response.result
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)
        
def arm_move(start_x, start_y, end_x, end_y):
    # 指令通过ros传输给机械臂
    from_x = base_x + increment * (start_x - 1) #base x/y/z 待定
    from_y = base_y + increment * (start_y - 1)
    to_x = base_x + increment * (end_x - 1)
    to_y = base_y + increment * (end_y - 1)

    # dobot_ptp_client(from_y, from_x, base_z, 0)
    # rospy.sleep(2)
        
    # # 启动吸盘以吸取物体
    # control_dobot_suction_cup(True, True)
    # rospy.sleep(1)  # 等待吸盘操作完成

    # # 移动到终止位置
    # dobot_ptp_client(to_y, to_x , base_z, 0)
    # rospy.sleep(4)
        
    # # 停止吸盘以释放物体
    # control_dobot_suction_cup(True, False)
    # rospy.sleep(1)  # 等待吸盘操作完成

    # dobot_ptp_client(to_y, to_x, 0, 0)
    # rospy.sleep(1)

    # dobot_home()
    
    return

def humoments(img_gray, log=True):
    """返回图像7个不变矩"""
    
    hu = cv2.HuMoments(cv2.moments(img_gray))[:,0]
    if log:
        hu = np.log(np.abs(hu))
    
    return hu


def lbp_histogram(img_gray, P=40, R=5, method='uniform'):
    """计算图像的LBP特征直方图"""
    lbp = local_binary_pattern(img_gray, P, R, method)
    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
    hist = hist.astype(np.float32)
    return hist

def jitter(im_cv, theta):
    """随机旋转和抖动，返回新的图像"""
    
    #theta = np.random.random()*360
    dx, dy = np.random.randint(0, 5, 2) - 2
    
    im_pil = Image.fromarray(im_cv)
    im_pil = im_pil.rotate(theta, translate=(dx, dy))
    im = np.array(im_pil)
    im[im==0] = 240
    im = np.where(im>104, 240, 15).astype(np.uint8) # 这里也是黑白二值化操作
    # im = cv2.adaptiveThreshold(piece, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # _, im = cv2.threshold(piece, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu's阈值
    return im

if __name__ == '__main__':
    # 定义一个96x96像素的掩码，用以滤除棋子周边的无效像素
    _mask = np.empty((96,96), dtype= bool)
    _mask.fill(False)
    for i in range(96):
        for j in range(96):
            if np.hypot((i-48), (j-48)) > 42:
                _mask[i,j] = True

    target = list() # 分类结果集
    data = list() # 样本数据集
    
    # chessman = ['車','马','炮','兵','卒','士','相','象','将','帅']
    chessman = ['車','炮','士','兵','相','马','帅']
    files = [
        ('/home/zhang/workspace01/ju.png', 0, 100),
        ('/home/zhang/workspace01/ju2.png', 0, 100),
        ('/home/zhang/workspace01/ju3.png', 0, 100),
        ('/home/zhang/workspace01/ju5.png', 0, 100),
        ('/home/zhang/workspace01/ju6.png', 0, 100),
        
        
        ('/home/zhang/workspace01/pao.png', 1, 100),
        ('/home/zhang/workspace01/pao2.png', 1, 100),
        ('/home/zhang/workspace01/pao3.png', 1, 100),
        
        ('/home/zhang/workspace01/shi.png', 2, 100),
        ('/home/zhang/workspace01/shi2.png', 2, 100),
        ('/home/zhang/workspace01/shi3.png', 2, 100),
        ('/home/zhang/workspace01/shi4.png', 2, 100),
        
        
        
        ('/home/zhang/workspace01/bing.png', 3, 100),
        ('/home/zhang/workspace01/bing2.png', 3, 100),
        ('/home/zhang/workspace01/bing3.png', 3, 100),
        
        ('/home/zhang/workspace01/xiang.png', 4, 100),
        ('/home/zhang/workspace01/xiang2.png', 4, 100),
        ('/home/zhang/workspace01/xiang3.png', 4, 100),
        
        ('/home/zhang/workspace01/ma.png', 5, 100),
        ('/home/zhang/workspace01/ma2.png', 5, 100),
        ('/home/zhang/workspace01/ma3.png', 5, 100),
        
        ('/home/zhang/workspace01/shuai.png',6, 100),
        ('/home/zhang/workspace01/shuai2.png',6, 100),
        ('/home/zhang/workspace01/shuai3.png',6, 100),
        ('/home/zhang/workspace01/shuai4.png',6, 100),
        ('/home/zhang/workspace01/shuai5.png',6, 100),
        ('/home/zhang/workspace01/shuai6.png',6, 100),
        
        
    ]
        
    for fn, idx, count in files:
        print('------------------------')
        print(fn)
        
        img = cv2.imread(fn)
        img_gray = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
        # img_gray = cv2.GaussianBlur(img_gray, (5,5), 0) # 首次高斯模糊
        # 圆检测
        circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 200, param1=150, param2=27, minRadius=30, maxRadius=65)
        circles = np.int_(np.around(circles))
        print(circles)
        for i, j, r in circles[0]:
            cv2.circle(img, (i, j), r, (0, 255, 0), 2)
            cv2.circle(img, (i, j), 2, (0, 0, 255), 3)
            
            # 图像预处理
            piece = cv2.resize(img_gray[j-r:j+r, i-r:i+r], (96,96))
            piece[_mask] = 240
            piece = cv2.GaussianBlur(piece, (5,5), 0) # 二次高斯模糊
            piece = np.where(piece>104, 240, 15).astype(np.uint8) # 黑白二值化阈值（3个地方）
            # piece = cv2.adaptiveThreshold(piece, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # 自适应阈值
            # _, piece = cv2.threshold(piece, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu's阈值
            # cv2.imwrite('res/chessman/%d_%d_%d.jpg'%(idx, i, j), piece)
            
            for theta in range(360):
                data.append(humoments(jitter(piece, theta))) # 如果使用不变矩
                # rotated_image = jitter(piece, theta)  # 假设这个函数可以旋转图像
                # lbp_hist = lbp_histogram(rotated_image)  # 计算LBP直方图
                # data.append(lbp_hist)
                target.append(idx)
            
         
        cv2.imshow('image', img)
        cv2.imshow('image_', piece)
        
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    target = np.array(target)
    data = np.stack(data, axis=0)
    print(data)
    print(data.shape)
    
    np.savez('cchessman.npz', target=target, data=data)

    ####################################################
    # 加载数据集
    ds = np.load('cchessman.npz')
    X = ds['data']
    y = ds['target']

    # 创建并训练随机森林分类器
    # classifier = RandomForestClassifier(criterion='entropy', max_depth=500, n_estimators=220, oob_score=True, min_samples_split=5)
    classifier = SVC(C=10, kernel='rbf', gamma=0.1)
    # classifier = MLPClassifier(hidden_layer_sizes=(500,), activation='relu', solver='adam',max_iter=2000, random_state=1)
    classifier.fit(X, y)

    # 加载并处理测试图像
    file_name = '/home/zhang/workspace01/all.png' #二次加载
    img = cv2.imread(file_name)
    img_gray = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # img_gray = cv2.GaussianBlur(img_gray, (5,5), 0) # 首次高斯模糊
    # 圆检测
    circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=21, minRadius=30, maxRadius=45)
    circles = np.int_(np.around(circles))
    print(circles)

    for i, j, r in circles[0]:
        # 绘制圆和中心点
        cv2.circle(img, (i, j), r, (0, 255, 0), 2)
        cv2.circle(img, (i, j), 2, (0, 0, 255), 3)

        # 处理图像
        piece = cv2.resize(img_gray[j-r:j+r, i-r:i+r], (96, 96))
        piece[_mask] = 240
        piece = cv2.GaussianBlur(piece, (5, 5), 0) # 二次高斯模糊
        piece = np.where(piece > 104, 240, 15).astype(np.uint8) # 黑白二值化
        # piece = cv2.adaptiveThreshold(piece, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2) # 自适应阈值
        # _, piece = cv2.threshold(piece, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu's阈值
        simples = []
        np.random.seed(42)
        for k in range(10):
            theta = np.random.random() * 360
            simples.append(humoments(jitter(piece, theta))) # 如果使用不变矩
            # rotated_image = jitter(piece, theta)  # 假设这个函数可以旋转图像
            # lbp_hist = lbp_histogram(rotated_image)  # 计算LBP直方图
            # simples.append(lbp_hist)
        simples = np.stack(simples, axis=0)

        # 预测分类
        y_pred = classifier.predict(simples)
        print(y_pred)
    
        print([chessman[k] for k in y_pred])

        # 使用字典来存储每个数字的出现次数
        num_count = {}
        for num in y_pred:
            if num in num_count:
                num_count[num] += 1
            else:
                num_count[num] = 1

        # 找出出现次数最多的数字
        most_common_num = max(num_count, key=num_count.get)
        print('The chess at (%f,%f) is %s'%(i,j,chessman[most_common_num]))
        pieces.append(ChessPiece(chessman[most_common_num],'黑',transform_pixel_to_digital_x(i),transform_pixel_to_digital_y(j)))
        print(transform_pixel_to_digital_x(i),transform_pixel_to_digital_y(j))
        # 保存并显示图像
        cv2.imwrite(f'res/test/{i}_{j}.jpg', piece)
        # cv2.imshow('image_gray', img)
        cv2.imshow('image', piece)
        cv2.imshow('image2',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()



# cannon1 = ChessPiece("炮", "红", 2, 3) # 初始化的xy坐标
# knight1 = ChessPiece("马", "红", 8, 1)
# knight2 = ChessPiece("马", "黑", 8, 9)
# pieces.append(cannon1)
# pieces.append(knight1)
# pieces.append(knight2)

rospy.init_node('dobot_control_node')

# 移动到起始位置
# start_x, start_y, start_z, start_r = 200, 0, -60, 0
# dobot_ptp_client(start_x, start_y, start_z, start_r)
# rospy.sleep(2)
    
# # 启动吸盘以吸取物体
# control_dobot_suction_cup(True, True)
# rospy.sleep(1)  # 等待吸盘操作完成

# # 移动到终止位置
# end_x, end_y, end_z, end_r = 250, 50, -60, 0
# dobot_ptp_client(end_x, end_y, end_z, end_r)
# rospy.sleep(4)
    
# # 停止吸盘以释放物体
# control_dobot_suction_cup(True, False)
# rospy.sleep(1)  # 等待吸盘操作完成

# dobot_ptp_client(end_x, end_y, 0, end_r)
# rospy.sleep(1)
dobot_home()
 
# rospy.init_node('testDobot')
#rospy.wait_for_service('DobotServer/SetHOMECmd')
# rospy.wait_for_service('DobotServer/SetEndEffectorSuctionCup')
# try:
#     set_home_client = rospy.ServiceProxy('DobotServer/SetHOMECmd',SetHOMECmd)
#     suction_client = rospy.ServiceProxy('DobotServer/SetEndEffectorSuctionCup', SetEndEffectorSuctionCup)
#     # response = dobot_home() # 回零
#     response = dobot_ptp_client(150,0,-56,0) # 移动至起始点
#     response = control_dobot_suction_cup(1, 1) # 吸
#     rospy.sleep(1)  # 等待吸盘动作完成
#     response = dobot_ptp_client(200,0,-56,0) # 移动至终止点（修改为终止点的坐标）
#     rospy.sleep(2)  # 等待动作完成
#     response = control_dobot_suction_cup(0, 0)# 释放物体
#     rospy.sleep(1)  # 等待释放动作完成
# except rospy.ServiceException as e:
#         print("Service call failed: %s"%e)
# result = dobot_ptp_client(200, 0, -20, 0)
while True:
    user_color = "黑"
    user_move = input("请输入象棋指令：")
    if user_move == "0":
        break
    uchess_move = chess_move(user_move)
    uchess_type = uchess_move[0]
    ucase = uchess_move[1] # 移动方式（平）
    ustart = uchess_move[2]
    uend = uchess_move[3]
    for item in pieces:
        if item.color == user_color:
            print(3)
            if item.chess_type == uchess_type: # item.chess_type == uchess_type
                print(item.x,ustart)
                if item.x == ustart:
                    print(2)
                    item_move = item.move(ucase, ustart, uend) # 返回起始、终止坐标 Eg.（1，1）
                    arm_move(item_move[0],item_move[1],item_move[2],item_move[3]) #执行机械臂移动
                    from_x = base_x + increment * (item_move[0] - 1) #base x/y/z 待定
                    from_y = base_y + increment * (item_move[1] - 1)
                    to_x = base_x + increment * (item_move[2] - 1)
                    to_y = base_y + increment * (item_move[3] - 1) 
                    print(from_x,from_y,to_x,to_y)
                    dobot_ptp_client(from_y, from_x, base_z, 0)
                    rospy.sleep(2)
                        
                    # 启动吸盘以吸取物体
                    control_dobot_suction_cup(True, True)
                    rospy.sleep(1)  # 等待吸盘操作完成

                    # 移动到终止位置
                    dobot_ptp_client(to_y, to_x , base_z, 0)
                    rospy.sleep(6)
                        
                    # 停止吸盘以释放物体
                    control_dobot_suction_cup(True, False)
                    rospy.sleep(1)  # 等待吸盘操作完成

                    dobot_ptp_client(to_y, to_x, 0, 0)
                    rospy.sleep(1)
                    print(1)
                    # 指令通过ros传输给机械臂
                else:
                    continue
            else:
                continue
        else:
            continue
    
for item in pieces:
    print(item.chess_type,item.x,item.y)