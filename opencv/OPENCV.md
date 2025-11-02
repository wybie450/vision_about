# 		***OPENCV***

！操作前导入库

<img src="OPENCV.assets/7df65bd8874dcb7e579f05ea20d410d9.jpg" alt="7df65bd8874dcb7e579f05ea20d410d9"  />

`import numpy as np / cv2 as cv / matplotlib.pyplot as plt`

## 一、openCV基本操作：

### 1. IO操作：

图像=二维数组

#### 1.1 读取

API  < `cv.imread('图像文件名'),读取方式`>

color彩色模式(1)   greyscale灰度模式(0)  unchanged加载图像模式(-1)

#### 1.2 展示

API <`cv.imshow(’窗口名‘,图片名)`>

cv.waitkey(0)

 #用matplotlib :	`plt.imshow(imag[:,:,::-1])`『bjr转rdp』

​	如果是灰度：`plt.imshow(img,cmap=plt.cm.grey)`

#### 1.3保存

API<`cv.imwrite('文件名',图像名)`>

### 2.绘制图形

#### 2.1直线

<`cv.line(img,start,end,color,thickness)`>

#### 2.2圆形

<`cv.circle(img,centerpoint,r,color,thickness)`>

#### 2.3矩形

<`cv.rectangle(ima,leftupper,rightdown,color,thickness)`>

#### 2.4添加文字

<`cv.putText(img,内容,字体,字号)`>

### 3.获取并修改图像像素点

bgr返回红绿蓝相对数组，灰度返回相应强<`blue=img[100,100,0]`>，修改像素值<`img[100,100]=[255,255,0]`>，获取像素点值<`px=img[100,100]`>

### 4.获取图像属性

<`img.size/shape/dtype`>数据类型

### 5.图像通道的拆分与合并

#### 5.1拆分

<`b,g,r = cv.split(img)`>

#### 5.2合并

<`img = cv.merge((b,g,r))`>

### 6.色彩空间的改变

bgr->Gray/HSV类型

<`X = cv.ctyColor(input_image,flag)`>

`flag=cv.COLOR_BGR2GREAY/COLOR_BGR2HSV`

##### ✴matplotlib和cv2的区别？

**(1) 图像颜色通道**

- OpenCV：

  默认使用 BGR 通道顺序。

- Matplotlib：

  默认使用 RGB通道顺序。

  - 如果直接用 `plt.imshow()` 显示 OpenCV 读取的图像，颜色会异常（红蓝通道互换）。

  - 解决方法：

    python

    复制

    ```
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 必须转换
    plt.imshow(image_rgb)
    ```

**(2) 交互性与事件处理**

- OpenCV：
  - 通过 `cv.waitKey()` 实现键盘交互（如按 `ESC` 退出）。
  - 支持鼠标事件回调（需 `cv2.setMouseCallback()`）。
- Matplotlib：
  - 内置工具栏（缩放、保存图像）。
  - 支持更复杂的事件绑定（如 `mpl_connect`）。

**(3) 性能与实时性**

- OpenCV：
  - 更适合 **实时视频处理**（如摄像头帧显示）。
  - 窗口轻量级，刷新速度快。
- Matplotlib：
  - 适合静态图像或数据分析可视化。
  - 渲染开销较大，实时性较差。

## 二、算术操作

### 1.图像的加法

opencv饱和操作：cv.add()

numpy模运算：a = x + y  『涉及位运算』

### 2.图像的混合

<`g(x) = a*f0(x) + b*f1(x)+c`> 『加权混合a+b=1』

<`img3 = cv.addweighted(img1,a,img2,b,c)`>

!要求两个图像同大小

## 三、图像处理

### 1.图像几何

#### 1.1图像缩放

<`cv2.resize(src,dsize,fx=0,fy=0,interpolation=cv2.INTER_LINEAR)`>

`interpolation=cv2.INTER_LINEAR/NEAREST/AREA/CUBIC`

#### 1.2图像平移

<`cv.warpAffine(img,M,(cols,rows))`>

rows 行数； cols 列数

`M =  np.float32([[x相关数据],[y相关数据]])`  创建M矩阵

#### 1.3图像旋转

需要修改每一个像素点坐标

<img src="OPENCV.assets/796d3494c655cc0f8fb8dd2204653d29.png" alt="796d3494c655cc0f8fb8dd2204653d29" style="zoom: 67%;" />

先根据旋转角度和中心得到旋转矩阵，再进行变换

API：`cv2.getRotationMatrix2D(center,angle,scale)`   

​		「scale为缩放比例」

​		 `cv.warpAffine(img,M,(cols,rows))`

#### 1.4仿射变换

概念为以上操作的组合，比例不变，角面关系不变

<img src="OPENCV.assets/48902fd2066f8243a73969d1789cc0f8.png" alt="48902fd2066f8243a73969d1789cc0f8" style="zoom: 50%;" />

API：

​		 `pts1=np.float32([[1点],[2点],[3点]])`

​		 `pts2=np.float32([[1点],[2点],[3点]])	`

​			「变化后」

​		 `cv2.getAffineTransform(pts1,pts2) `

​			「scale为缩放比例」

​		 `cv.warpAffine(img,M,(cols,rows))`

#### 1.5透射变化

视角变化的结果，存在三点共线，类似投影一个图像到新的面上<img src="OPENCV.assets/d666418620366b6b781cb6128325dd4f.png" alt="d666418620366b6b781cb6128325dd4f" style="zoom:50%;" />

​		 `pts1=np.float32([[1点],[2点],[3点],,[4点]])`

​		 `pts2=np.float32([[1点],[2点],[3点],,[4点]])`

   		「变化后」

​		 `cv2.getPerspectiveTransform(pts1,pts2)`  

​		 「scale为缩放比例」

​		 `cv.warpPerspective(img,M,(cols,rows))`

#### 1.6图像金字塔

同一张原始图片不断向上采样，分辨率逐级降低，到阈值停止

API：	`cv.pyrUp(img)`

​			  `cv.pyrDown(img)`

### 2.形态学操作

#### 2.1连通性

像素点附近有三种形式的邻接像素：

4邻接[(x,y-1),(x,y+1),(x+1,y),(x-1,y)]、D邻接[(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]、8邻接[4+D]

<img src="OPENCV.assets/cd08eb716642356ef008aea83a854cf8.png" alt="cd08eb716642356ef008aea83a854cf8" style="zoom: 67%;" />

连通条件：相邻；灰度相似

连通种类：4连通；8连通

m连通条件：q在ND(p)中，且N4(p)，N4(q)两种交集无灰度为V的像素「即p、q本身」

#### 2.2腐蚀和膨胀

##### 2.2.1.腐蚀：

用结构元素“与”图像中每一个像素，求局部最小值

✴腐蚀后，只有结构元素中每个元素都为1，结构A中间的元素才会保留，否则为0，目的消除噪点

<img src="OPENCV.assets/d575dd2cbf9af745f077e9280050a1c5.png" alt="d575dd2cbf9af745f077e9280050a1c5" style="zoom: 50%;" />

API：`cv.erode(img,kernel,iterations)` 

​		 「图像、核结构、腐蚀次数」

##### 2.2.2.膨胀：

用结构元素“或”图像中每一个像素，求局部最大值

✴原理同上，目的填补孔洞

API：`cv.dilate(img,kernel,iterations)` 

​		 「图像、核结构、腐蚀次数」

​		 `kenel=np.ones((x,y),np.uint8)`	「创建核结构」

###### ✴plt图像展示相关

<img src="OPENCV.assets/898e8c38019c41f77e45232f74b0d3a0.png" alt="898e8c38019c41f77e45232f74b0d3a0" style="zoom: 67%;" />

`fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 8), dpi=100)`
创建一个 1 行 3 列 的子图（共 3 个小图）；

- figsize=(10,8)：设置整个画布尺寸为 10×8 英寸；


- dpi=100：图像分辨率为 100 像素/英寸；


- axes 是包含 3 个子图坐标轴的数组（axes[0], axes[1], axes[2]）。

#### 2.3开闭运算

##### 2.3.1.开运算：

先腐蚀后膨胀，目的减少噪点，不影响原图像

##### 2.3.2.闭运算：

先膨胀后腐蚀，目的填充闭合区域

API：`cv.morphologyEx(img,op,kernel)`

op->`cv.MORPH_OPEN`「开」   `cv.MORPH_CLOSE`「关」

#### 2.4黑帽与礼帽

2.4.1.礼帽运算：原图像与开运算之差，目的突出原图轮廓更明亮的区域

dst=tophat(src,element)=src-open(src,element)

✴它只能用于分离邻近的，有大幅背景且物体有规律时可用顶帽运算

2.4.2.黑帽运算：闭运算和原图像之差，目的突出更暗区域

dst=blackhat(src,element)=close(src,element)-src

API:`cv.morphologyEx(img,op,kernel)`

op->`cv.MORPH_TOPHAT`「开」   `cv.MORPH_BLACKHAT`「关」

✴它们都只适用于邻近点附近

### 3.图像平滑

#### 3.1图像噪声

##### 3.1.1.椒盐噪声：

也叫脉冲噪声，随机出现的白点和黑点

##### 3.1.2.高斯噪声：

噪声密度函数服从高斯分布

#### 3.2图像平滑

##### 3.2.1.均值滤波：

用滤波模板对噪声进行滤波，即卷值框覆盖区平均值代替中心元素

✴用法简单但易使图像模糊，模糊但不去除就是核小了

API:  `cv.blur(src,ksize,anchor,borderType)`  

​		「图像，卷积核，核中心，边界类型」

##### 3.2.2.高斯滤波：

根据高斯分布中钟形曲线分配权重，用加权平均值乘上原像素灰度值

确定权重矩阵->再将8邻接值相加->得到高斯模糊的值![1c98a2a0970d3df3adad88a17158ce16](OPENCV.assets/1c98a2a0970d3df3adad88a17158ce16.png)

✴如果原图是彩色，就对RGP三个通道做高斯平滑

API： `cv2,GaussianBlur(src,ksize,sigmaX,sigmay,borderType)`

​		「图像，卷积核[奇数]，水平标准差，竖直标准差，边界类型」

##### 3.3.3.中值滤波：

用像素点领域灰度值中值代替像素的灰度值

✴对椒盐噪声有用

API: ` cv.medianBlur(src,ksize)`

### 4.直方图

✴传入参数要用中括号，由API决定

#### 4.1灰度直方图

表示图像亮度分布，其根据灰度值绘制

转换为灰度图->分割数字范围为子区域->统计bin(i)像素数目

- dims:  需要统计的特征数目
- bins:  特征空间子区段数目
- range:  统计特征的取值范围

意义：

- 像素强度的图形表达方式
- 统计每一个强度值所具有像素点个数
- 不同图像直方图可能一样

#### 4.2直方图的计算和绘制

API:  `cv2.calcHist(img,Channels,mask,histSize,ranges[,hist[,accmulate]])`  用一个值取出来hist

mask->掩模图图像	histSize->BIN数目	ranges->像素值范围

#用matplotlib :

`plt.figure(figsize=(10.8))`创建画布

`plt.plot(hist)`画折线

`plt.grid()`转换形式

`plt.show()`展示

#### 4.3掩膜的应用

用选定东西对图像进行遮挡，控制图像处理区域。常为二位矩阵数组<img src="OPENCV.assets/b3fed82275ecdb2dc1afab2c35a11529.png" alt="b3fed82275ecdb2dc1afab2c35a11529" style="zoom:67%;" />

AIP：`mask_img=cv.bitwise_and(img,img,mask=mask)`

​			「两个图像‘与’操作」

#### 4.4直方图均衡化

当很多像素点都在一个很小的灰度值内，将它的直方图拉伸，重新分配像素值，增大对比度，即直方图均衡化

目的解决曝光过度或曝光不足

API：`dst=cv.equaliseHist(img)`

#### 4.5自适应均衡化

将图像分成多个小块，分别进行均衡化，为了避免噪声，用对比度限制，当他的bin超过对比度上限，就将它均衡到附近的bin，最后用双线性差值拼接每一小块

API:	`clahe=cv.createCLAHE(clipLImit,tileGridSize)`

- clipLimit:对比度限制
- tileGridSize:分成x*y小块

​		`cl1=clahe.apply(img)`	应用于图像

### 5.边缘检测

标识图像中亮度变化明显的点

- 基于搜索：搜索一阶导数的最大值检测边界，计算估计边缘方向，采用梯度找到模的最大值，代表算法Sobel,Scharr
- 基于零穿越：搜索二阶导数零穿越来寻找边界，代表算法Laplacian算子

#### ![01fb89a860de276981dd4d0e4d0201ea](OPENCV.assets/01fb89a860de276981dd4d0e4d0201ea-17509077286321.png)5.1Sobel算子

效率高，检测边缘不准确

✴内核大小为3时，用Scharr函数更准确

API：`Sobel_x_or_y=cv2.Sobel(src,ddepth,dx,dy,dst,ksize,scale,delta,borderType)`

- ddepth:图像深度
- dx,dy:求导阶数
- ksize:Sobel算子大小![fe2c30e0b5f6594da6cc78fff5e60f3e](OPENCV.assets/fe2c30e0b5f6594da6cc78fff5e60f3e.png)

算子两个方向上计算，需要最后组合

`Scale_abs=cv2.convertScaleAbs(x) #格式转换`

`result=cv2.addweight(src1,alpha,src2,beta) #图像混合`

✴Sober算子部分的ksize设为-1，就是利用Scharr算法

#### 5.2Laplacian算子

二阶导数检测算子

API:  `laplacian=cv2.Laplacian(src,ddepth[,dst[,ksize[,scale[,delta[,borderType[)`

✴ksize为奇数

###### ✴plt图像展示相关

![8e00be65bb7cd04b614eab2ee3c68160](OPENCV.assets/8e00be65bb7cd04b614eab2ee3c68160.png)

- plt.subplot() ：创建小窗
- plt.xticks() : 确定x光轴位置

#### 5.3Canny边缘检测

噪音去除（5*5）->计算图像梯度（Sober）->非极大值抑制（去除非边界点）->滞后阈值（确定边界像素点灰度值需高于maxVal，若不高于最大也不低于最小，就检测是否与确实边界点相连）

API:  `canny=cv2.Canny(img,threshold1,threshold2)`

- threshold1:  较小阈值
- threshold2:  较大阈值

### 6.模板匹配和霍夫变换

#### 6.1模板匹配

在给定图片中查找与模板最相似的区域（逐个像素点计算）

API:  `res=cv.matchTemplate(img,template,method)`

- template:  要匹配的模板

- method:  计算相似的算法

  ​	1.平方差匹配(`CV_TM_SQDIFF`)：0最好，越大越差

  ​	2.相关匹配(`CV_TM_CCORR)`：两者相乘，越大越好

  ​	3.利用相关系数匹配(`CV_TM_CCOEFF`)：1完美匹配，-1最差

 完成匹配后，`cv.minMaxlog()`查找最大值所在位置，平方差查最小<img src="OPENCV.assets/1bd47d520d6fe814068a68f5f467219c.png" alt="1bd47d520d6fe814068a68f5f467219c" style="zoom:67%;" />

得到返回值后绘制图像

！模板匹配不适用于尺度变换，视角变换的图像，需用关键点匹配法，相关算法有：SIFT,SURF

#### 6.2霍夫变换

用于提取图像中圆和直线等几何形状

原理： 直线由(k,q)变化成一个点后的坐标称为霍夫空间，反过来霍夫空间的线也为笛卡尔坐标系的点，当有多个点时，选择霍夫空间中相交最多次的点，无交点（k相同）时，笛卡尔坐标系变为极坐标系，用上面的点变为霍夫空间的线

##### 6.2.1霍夫线检测

API：`cv.HoughLines(img,rho,theta,threshold)`

- img:二值化图像

- rho,theta:极坐标参数精度（累加器行列）

  `np.pi/180`即一度的精度

- threshold:阈值，累加器超过它才会认为是直线

<img src="OPENCV.assets/3a8acd307b59d50a487ea28083d4b065-17509206759092.png" alt="3a8acd307b59d50a487ea28083d4b065" style="zoom: 50%;" />

绘制直线：

<img src="OPENCV.assets/5a1121eb07c54b159a8a9295ca6c5b50.png" alt="5a1121eb07c54b159a8a9295ca6c5b50" style="zoom:67%;" />

！x0,y0确定极坐标点后，后面的用于绘制穿过整个图片的直线，注意是整数类型

##### 6.2.2霍夫圆检测

霍夫梯度法：

目的减少霍夫空间维度，减小效率

检测圆心->推导半径，原理都为设置阈值，符合条件的直线数量超过则可以确定

API:  `circles=cv.HoughCircles(img,method,dp,minDist,param=100,param=100,minRadius=0,minRadius=0,maxRadius=100)`![6e4b6067527ddd91ca2fffe899a93a89](OPENCV.assets/6e4b6067527ddd91ca2fffe899a93a89.png)

## 四.角点特征

图像特征中的角点、斑点都是很好识别的

### 1.Harris/Shi-Tomasi算法

#### 1.1Harris角点检测

<img src="OPENCV.assets/fea71a6dbb1404c1a0940b3a4e8b3675.png" alt="fea71a6dbb1404c1a0940b3a4e8b3675" style="zoom: 50%;" />

- w(x,y):表示加权，可以矩形加权区分窗口内外，还有高斯加权

- x,y的灰度值变化构成一个椭圆函数，由下识别![31c43b96a3127e009c7c3063cf5f22a1](OPENCV.assets/31c43b96a3127e009c7c3063cf5f22a1.png)

- Harris用矩阵行列式和迹相减整合成R,如下表示<img src="OPENCV.assets/37b1eb873f60ceae1e5431d92963cf18.png" alt="37b1eb873f60ceae1e5431d92963cf18" style="zoom: 25%;" />

“Flat”是R为正值但很小

API:  `dst=cv.cornerHarris(src,blockSize,ksize,k)`<img src="OPENCV.assets/8486f16fe94681d07d8e6b486cf119fb.png" alt="8486f16fe94681d07d8e6b486cf119fb" style="zoom: 67%;" />

!最后一个参数在0.04～0.05间

！计算前用`np.float32()`转换数据类型

#### 1.2Shi-Tomasi角点检测

他们用矩阵特征值较小的一个大于阈值，则认为它是角点<img src="OPENCV.assets/59c97ca6b43e506ffe7a41866aeec979.png" alt="59c97ca6b43e506ffe7a41866aeec979" style="zoom: 50%;" />

API:  `corners=cv.goodFeaturesToTrack(img,maxcorners,qualitylevel,minDistance)`

<img src="OPENCV.assets/e716e484b2050f6a53b1c6698bb93cde.png" alt="e716e484b2050f6a53b1c6698bb93cde" style="zoom:50%;" />

### 2.SIFT/SURF算法

角点检测具有旋转不变性，但没有尺寸不变性

此算法实质是在图像上查找关键点，具有位置，尺度，旋转不变性

空间极值检测->关键点定位->方向确定->关键点描述

#### 2.1SIFT算法

##### 2.1.1尺度空间极值检测

运用高斯核，小关键点用小窗口，大关键点用大窗口![67b4fb47d7f8799d0518ebe5a65f15b8](OPENCV.assets/67b4fb47d7f8799d0518ebe5a65f15b8.png)

![image-20250626163054809](OPENCV.assets/image-20250626163054809.png)

！金字塔6层，相减后-1，对比时-2，留三层找最大值



得到局部极值点后还要与阈值对比，若灰度值小于则去掉，此法能去掉对比度低像素点和边界

##### 2.2.2关键点方向确定

有了尺寸不变性后，为了实现旋转不变性，得出关键点所在高斯金字塔上以r为半径的梯度特征，在其领域分配方向角度

<img src="OPENCV.assets/2abee1e60f2176e5af15df2d496717c7.png" alt="2abee1e60f2176e5af15df2d496717c7" style="zoom: 50%;" />

每个特征点除了一个主方向，幅值大于其80%的可以作辅方向<img src="OPENCV.assets/32a3fe1c1bd52a3260a988a12221c9f5.png" alt="32a3fe1c1bd52a3260a988a12221c9f5" style="zoom:50%;" />

再进行以下步骤：

<img src="OPENCV.assets/d462a75061db4b95839e4998af797bb0-17509274621064.png" alt="d462a75061db4b95839e4998af797bb0" style="zoom:50%;" />

为每个关键点建立一个描述符：通过关键点周围图像分块，计算块内梯度直方图，生成特征向量，对图像信息进行抽象<img src="OPENCV.assets/36654ec9bd77de2b5e46e50f7bd7e09a.png" alt="36654ec9bd77de2b5e46e50f7bd7e09a" style="zoom:50%;" />

求得最终累积到每个方向的梯度，同理求得4x4x8=128个的梯度信息即为该关键点的特征向量

#### 2.2SURF算法

SIFT存在实时性不高，特征点少且不准确的缺点，SURF算法是对其的优化算法，速度更快<img src="OPENCV.assets/cb4cc3b7bf62a7fc80acc48900b0ac9d.png" alt="cb4cc3b7bf62a7fc80acc48900b0ac9d" style="zoom: 50%;" />

#### 2.3SIFT实现

<img src="OPENCV.assets/f1ce59b53d3092d72e64f3d2bf784b40.png" alt="f1ce59b53d3092d72e64f3d2bf784b40" style="zoom: 50%;" />

<img src="OPENCV.assets/87603df32ef19af2af173a68cabfd9f0.png" alt="87603df32ef19af2af173a68cabfd9f0" style="zoom: 50%;" />

### 3.Fast和ORB算法

#### 3.1Fast算法

目的从实时性出发，提高检测角点效率

<img src="OPENCV.assets/237478a04ea1e4a66442f34f92fe3a11.png" alt="237478a04ea1e4a66442f34f92fe3a11" style="zoom:50%;" />

准确性和效率可以通过模型训练提升，相邻特征点的问题可以通过非最大值抑制的方法解决

##### 3.1.1机器学习的角点检测器<img src="OPENCV.assets/c76fb87c8eff941346a43c9090d66921.png" alt="c76fb87c8eff941346a43c9090d66921" style="zoom:50%;" />

<img src="OPENCV.assets/c27f206526e1625a43779913170dd288.png" alt="c27f206526e1625a43779913170dd288" style="zoom: 50%;" />

##### 3.1.2非极大值抑制

<img src="OPENCV.assets/4a6249dee7207567e78de4a98929502d.png" alt="4a6249dee7207567e78de4a98929502d" style="zoom:50%;" />

##### 3.1.3实现

1. 实例化fast

   `fast=cv.FastFeatureDetector_create(threshold,nonmaxSupporession)`

   <img src="OPENCV.assets/2d7db137a10759ad1b7a5768fa4fa035.png" alt="2d7db137a10759ad1b7a5768fa4fa035" style="zoom:50%;" />

​	2.利用fast.detect检测关键点，没有相应关键点描述

​		`kp=fast.detect(greyImg,None)`

​		✴图片类型可以是彩色

​	3.将关键点检测结果绘制到图像上

`cv.drawKeypoints(img,keypoints,outputimage,color,flags)`

->默认下打开非极大抑制，要关闭的话可以使用`fast.setNonmaxSuppression(0)`

`kp=fast.detect(img,None)`

#### 3.2ORB算法

两个S算法要钱，这个免费；可以用于快速创建关键点特征变量<img src="OPENCV.assets/5df5f3e7650bb5ad41befd729d558feb.png" alt="5df5f3e7650bb5ad41befd729d558feb" style="zoom:50%;" />

<img src="OPENCV.assets/0201678931049da920aaf4bad21a19d7.png" alt="0201678931049da920aaf4bad21a19d7" style="zoom:50%;" />

将特征点领域旋转到主方向上后用Brief算法构建描述符

##### 3.2.1Brief算法<img src="OPENCV.assets/a377e1a42b46681fd8f59f4abba0e4b3.png" alt="a377e1a42b46681fd8f59f4abba0e4b3" style="zoom: 50%;" />

<img src="OPENCV.assets/0614accd6a2fe5ec5ec94cb43c2ddd17.png" alt="0614accd6a2fe5ec5ec94cb43c2ddd17" style="zoom:50%;" />

选取时是极坐标系形式

##### 3.2.2实现

​	1.实例化ORB

​	`orb=cv.xfeatures2d.orb_create(nfeatures)`

- nfeatures:  特征点的最大数量

​	2.检测关键点并计算

​	`kp,des=orb.detectAndCompute(grey,None)`

​	3.将结果绘制在图像上

​	代码同其他算法

## 五、视频操作

### 1.视频读写

#### 1.1读取与显示

​	1.创建读取视频的对象

​	`cap=cv.VideoCapture(filepath)`

​	2.视频的属性信息

​	2.2获取属性

​	`retval=cap.get(propId)`

<img src="OPENCV.assets/5864799c4b1854535352426cc0fa11ad.png" alt="5864799c4b1854535352426cc0fa11ad" style="zoom:50%;" />

​	2.3修改属性信息

​	`cap.set(propId,value)`

- propId:  属性索引
- value:  修改后的属性值

​	3.判断是否读取成功

​	`isornot=cap.isOpened()`

- 读取成功则返回true

​	4.获取视频的一帧图像

​	`ret,frame=cap.read()`

<img src="OPENCV.assets/05d7907cf2fccdf8f04302eb1567c05a.png" alt="05d7907cf2fccdf8f04302eb1567c05a" style="zoom: 50%;" />

### 2.视频保存

<img src="OPENCV.assets/0cb314e78ef2af00a281dd9b94f0f424.png" alt="0cb314e78ef2af00a281dd9b94f0f424" style="zoom:50%;" />

<img src="OPENCV.assets/876dbe1960b6ad3be61388160a627959.png" alt="876dbe1960b6ad3be61388160a627959" style="zoom:50%;" />

### 3.meanshift原理<img src="OPENCV.assets/e304940f705b411bb37c715e6182447c.png" alt="e304940f705b411bb37c715e6182447c" style="zoom:50%;" />

<img src="OPENCV.assets/8edc5bc326a5f4bb543b0dbda05c9a3e.png" alt="8edc5bc326a5f4bb543b0dbda05c9a3e" style="zoom:50%;" />

#### 3.1实现

API:  `cv.meanShift(probImage,window,criteria)`<img src="OPENCV.assets/711a76b4ae6ce4681b8d4067c942b5fb.png" alt="711a76b4ae6ce4681b8d4067c942b5fb" style="zoom:50%;" />

实例：

<img src="OPENCV.assets/793af7e19bb43366ac1ae187dd71bb04.png" alt="793af7e19bb43366ac1ae187dd71bb04" style="zoom: 67%;" />

<img src="OPENCV.assets/c38c18ac5225bfa4d0083624257fb1e4.png" alt="c38c18ac5225bfa4d0083624257fb1e4" style="zoom:67%;" />

✴r:r+h:  意为r到r加上视频的高

### 4.Camshift算法

检测窗口大小固定，不适用由近及远，此算法可随着跟踪目标大小实时调整搜索窗口大小

#### 4.1实现

<img src="OPENCV.assets/16b95e5b4a20d402b63fbb1a4b2415b3.png" alt="16b95e5b4a20d402b63fbb1a4b2415b3" style="zoom:67%;" />

缺点：背景色和目标颜色相近时易使目标区域变大，导致目标跟踪丢失

## 六、人脸检测

### 1.基础

<img src="OPENCV.assets/67bd7e0b2b76adbc39f9eabbde4ad474.png" alt="67bd7e0b2b76adbc39f9eabbde4ad474" style="zoom:67%;" />

<img src="OPENCV.assets/783747c75aa61ea8c4d5e2ddcc22cd20.png" alt="783747c75aa61ea8c4d5e2ddcc22cd20" style="zoom:67%;" />

<img src="OPENCV.assets/0ebcbf4d555cd92d8f1f815368136452.png" alt="0ebcbf4d555cd92d8f1f815368136452" style="zoom:50%;" />

### 2.实现

<img src="OPENCV.assets/f2bb4ef6c35071725827cbe8d42c0acf.png" alt="f2bb4ef6c35071725827cbe8d42c0acf" style="zoom: 67%;" />

#### 2.1视频的人脸检测

<img src="OPENCV.assets/37a2a85eaac09a6544f43d94a9f09a14.png" alt="37a2a85eaac09a6544f43d94a9f09a14" style="zoom: 67%;" />

->`pkg-config --cflags --libs opencv4`
这个是找路径的命令

###### ✴查找电脑连接摄像头设备

 `v4l2-ctl --list-devices`

## 扫盲

###### 什么是blobFromImage函数？

https://blog.csdn.net/wxy2020915/article/details/126749072

##### 附加：

有时候跑python在ros的环境下不兼容，可以通过python3创建独立的新环境

1. 创建`python3 -m venv venv`
2. 启动虚拟环境`source venv/bin/activate`
3. 退出虚拟环境`deactivate`