# 结合pyqt5部署【AI骨龄计算项目】

## 项目描述
骨龄计算是一项常规放射X线检查项目。是一种得出儿童的【生长年龄】简单有效的方法。对儿童的生长发育情况做出有效的判断。但是，它却是一项耗时耗精力的活儿。在前一段时间，出过一个项目[【骨龄计算综合应用：使用飞桨让医生再腾出10分钟】](https://aistudio.baidu.com/aistudio/projectdetail/1485230) ，发现使用深度学习中的检测+分类，可以有效帮助医生自动化计算骨龄。
现在结合PyQt5制作桌面应用，真正落地到医生的诊断电脑上。很方便的获取医学影像PACS系统上的手部X光图，只需要4到5秒即可得出骨龄结果。
项目设计时增加了修改功能。医生对AI预测的结果随时修改，回归到医生下诊断，AI做辅助的状况。最终得到一份有临床价值的骨龄诊断报告。
从本来需要10分钟的才能出一份骨龄报告，现在只需要几秒。

【用到框架】图像界面：pyqt5，深度框架paddlepaddle

## 项目结构
```
-|config  #存放设置配置文件
-|image   #存放图谱
-|inference_model  #存放模型文件
-|boneagehtml   #存放图谱网页
-|html  #介绍项目的网页
-boneMain.py  #程序主文件
-config.py    #骨龄标准分等
-preprocess.py  #图像处理，resize等，对需要目标检测的图片进行数据预处理
-UI2.py    #UI文件
-UI2.ui   #UI文件
-utils.py  #一些重要的工具类
-README.MD
-main.ipynb
```
## 使用方式
A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/usercenter)
B：项目运行的环境:win64，python3.6，PaddlePaddle2.0  
