#-*-coding: utf-8-*-

import  os 
import  shutil 
import argparse
import parser
import cv2
import numpy as np  

# img_dir="/home/deakin/Desktop/fingers/sdu01/train/finger01"
# img_dir="/home/deakin/Desktop/fingers/my/train/finger1"
img_dir="/media/deakin/00000354000386C6/work/finger-vein/finger-17-50/train/00000"

###加载图像，获取ROI
# class ROI_Get:
    # def __init__( self ):
###提取静脉ROI
# 传入原始图片，需要为标准灰度图
def get_ROI( img , f_threshold  ):
    #二值化操作
    #设定二值化阈值,后续需要改为动态阈值
    _, thresh = cv2.threshold(  img , thresh=f_threshold, maxval=255, type=0  )
    # 获取最大轮廓 , 进一步提取ROI区域 
    contours, hierarchy = cv2.findContours(  thresh  ,  cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE  )
    print( "contours size:", contours.__len__() )
    # 判断个数是否大于等于1
    if( contours.__len__()  <  1 ):
            return None
    #保存区域面积
    areas = np.array([])  
    for conto in contours:
        # areas.append ( conto.__len__() )
        areas = np.append( areas, conto.__len__( ), axis=None)
    print("areas: ", areas )
    max_index =  np.argmax( areas ) 
    print("max_index", max_index )
    x_all = np.array([], dtype=int )  
    y_all = np.array([], dtype=int )  
    for p in contours[ max_index ]:
        # print( p )
        cv2.circle(  thresh , ( p[0][0], p[0][1]), 5, 200, 0  )
        x_all = np.append( x_all , p[0][0] )
        y_all = np.append( y_all , p[0][1] )
    ##找最大最小值
    x_min = x_all.min( )
    y_min = y_all.min( )
    x_max = x_all.max( )
    y_max = y_all.max( )
    # print("x_min:{} y_min:{} ", x_min, y_min )
    # print("x_max:{} y_max:{} ", x_max, y_max )
    rect = np.array( [ x_min, y_min, x_max - x_min, y_max - y_min  ], dtype=int )
    print( rect )
    return thresh,rect
"""
if __name__ == "__main__":
    print("============ROIGET===============")
    base_index = 0
    for index  in range( 50 ):
        file_name = "/finger_" +  str( base_index + index ).zfill( 8 ) + ".jpg"
        img_path = img_dir + file_name
        print("image path:", img_path )
        img = cv2.imread( img_path , cv2.IMREAD_GRAYSCALE )  ###jpg转为灰度
        if( img.any() == None ):
            print("load image fail!")
            exit(0)
        ##进行ROＩ提取
        thresh, rect = get_ROI( img, 30 )
        ##裁剪ROI区域
        # y:y+h x:x+w
        roi = img[ rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2] ] 
        # equ = cv2.equalizeHist( roi )

        cv2.imshow("origin", img )
        cv2.imshow("thresh", thresh  )
        # cv2.imshow("equ", equ  )
        cv2.imshow("roi", roi  )
        cv2.waitKey(100)
        input(">>>push enter  to next.")

    cv2.destroyAllWindows()

"""

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--files_path", type=str, default="images/", help="path for  images files")
    parser.add_argument("--cut_files_path", type=str, default="cut_images/", help="path for  images  after cut")

    params_opt = parser.parse_args()

    ###开始遍历数据文件夹
    for parent, dirs, file_names in os.walk( params_opt.files_path, followlinks=True ):
        for file_name in file_names:
            file_path = os.path.join( parent,  file_name )
            print("parent: {}".format( parent ) )
            print('full path : {}'.format(  file_path ) )
            img_path = file_path
            ## 1.提取roi区域
            """
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # jpg转为灰度
            if(img.any() == None):
                print("load image fail!")
                exit(0)
            ##进行ROＩ提取
            thresh, rect = get_ROI( img, f_threshold=30 )
            ##裁剪ROI区域
            # y:y+h x:x+w
            roi = img[ rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2] ] 

            ###修改尺寸，双线性插值
            roi = cv2.resize(  roi, ( 320, 128 ), interpolation=cv2.INTER_LINEAR  )  
            ###保存ROI图像
            new_img_path =  img_path[0:-3] + "bmp"  ##修改后缀
            cv2.imwrite( new_img_path , roi )

            # cv2.imshow("origin", img )
            # cv2.imshow("roi", roi  )
            # cv2.waitKey(100)
            # input(">>>push enter  to next.")
            """
            ##2.删除后缀为jpg的文件
            if( img_path[-3:] == "jpg" ):
                os.remove( file_path )
            """
            """
        print("一共写入{}个数据, 保存在目录{}".format( file_names.__len__()  ,  parent   ) )
    cv2.destroyAllWindows()


