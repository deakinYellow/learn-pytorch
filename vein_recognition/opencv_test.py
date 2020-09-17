#-*-coding: utf-8-*-
import json
import cv2
import numpy as np  
import torch

# img_dir="/home/deakin/Desktop/fingers/sdu01/train/finger01"
# img_dir="/home/deakin/Desktop/fingers/my/train/finger1"
img_dir="/media/deakin/00000354000386C6/work/finger-vein/finger-17-100-cl/00000"

if __name__ == "__main__":
    print("============ROIGET===============")
    # img_path = img_dir + "/index_1_00000001.bmp"
    # img_path = img_dir + "/index_2_00000002.bmp"
    # img_path = img_dir + "/index_3_00000003.bmp"
    # img_path = img_dir + "/index_4_00000004.bmp"
    img_path = img_dir + "/finger_00000000.jpg"

    img = cv2.imread( img_path , cv2.IMREAD_GRAYSCALE )  ###jpg转为灰度
    if( img.all() == None ):
        print("load image fail!")
        exit(0)
    ##直方图均衡化
    equ = cv2.equalizeHist( img )

    ##上下翻转
    filp_img = cv2.flip( img, 0 )
    canny = cv2.Canny(img, 10, 50)

    _, thresh = cv2.threshold( img , thresh=30, maxval=255, type=0  )

    ##sobel
    # Sobel(src, ddepth, dx, dy[, dst[, lljjksize[, scale[, delta[, borderType]]]]]) -> dst
    ##对y方向求梯度
    ###将原图像分为上下两张
    # img_up = cv2.

    y = cv2.Sobel(  img, cv2.CV_8U, 0, 1,  ksize= 7   )

    flip_y = cv2.Sobel( filp_img , cv2.CV_8U , 0, 1,  ksize=3  )
    flip_y_f = cv2.flip( flip_y, 0 )

    _, thresh_1 = cv2.threshold( y , thresh=50, maxval=255, type=0  )
    _, thresh_2 = cv2.threshold( flip_y_f , thresh=220, maxval=255, type=0  )

    sobel_img = cv2.addWeighted(  thresh_1 ,1 ,   thresh_2 ,  1,  0 )
    ##形态学滤波
    # 闭操作对小区域进行联通
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT,(8, 4))
    closed = cv2.morphologyEx( sobel_img , cv2.MORPH_CLOSE, kernel_close )

    #开运算
    # kernel_open = cv2.getStructuringElement( cv2.MORPH_RECT,(3, 1))
    # opened = cv2.morphologyEx( closed  , cv2.MORPH_OPEN, kernel_open )
    ##基尔霍夫直线检测
    """
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP( sobel_img , 1, np.pi / 30, 100, minLineLength, maxLineGap )
    print("lines size: {}".format( lines.__len__( ) ) )
    for x1, y1, x2, y2 in lines[0]:
        cv2.line( sobel_img , (x1, y1), (x2, y2), 150 , 2)
    """

    # 获取最大轮廓
    # findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
    # contours, hierarchy = cv2.findContours( closed ,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )
    contours, hierarchy = cv2.findContours( closed ,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE  )
    # print( "hierarchy:", hierarchy )
    print( "contours size:", contours.__len__() )
    areas = np.array([])
    # 判断个数是否大于等于2
    for conto in contours:
        # areas.append ( conto.__len__() )
        areas = np.append( areas, conto.__len__( ), axis=None)

    print("areas: ", areas )
    print("areas max index: ", np.argmax( areas )  )

    """
    max_index_1 =  np.argmax( areas ) 
    areas = np.delete( areas, max_index_1 )
    max_index_2 =  np.argmax( areas ) 

    if( max_index_2 >= max_index_1 ):
        max_index_2 = max_index_2  + 1 
    print("areas max index 1: {}  2: {}  : ".format(  max_index_1, max_index_2 ) )

    for p in contours[ max_index_1 ]:
        # print( p )
        cv2.circle( img, ( p[0][0], p[0][1]), 5, 230, 0  )

    for p in contours[ max_index_2 ]:
        # print( p )
        cv2.circle( img , ( p[0][0], p[0][1]), 5, 230, 0  )
    """

    cv2.imshow("origin", img )
    cv2.imshow('sobel_y',  y )
    # cv2.imshow('sobel_y flip',  flip_y_f  )
    # cv2.imshow('canny', canny  )
    cv2.imshow('equ',  equ )
    cv2.imshow('thresh',  thresh )
    # cv2.imshow('sobel',  sobel_img )
    # cv2.imshow('close', closed )
    # cv2.imshow('thresh_open',  opened )

    cv2.waitKey(100)
    input(">>>>>>>>>>>>")
    cv2.destroyAllWindows()

