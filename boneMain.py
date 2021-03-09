import sys
import ctypes
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtWidgets
from UI2 import Ui_MainWindow
import os
import paddle
import numpy as np
import cv2
from preprocess import *
from utils import Detector, Config,  calcBoneAge,CaptureScreen,MyQLabel
from config import SCORE,Arthrosis
import pysnooper
import pyperclip
import time
import json
import webbrowser as web
import win32con
"""重载Qlabel控件"""
QtWidgets.QLabel = MyQLabel

class InitModelThread(QThread):  # 建立一个任务线程类
    """
    初始化模型
    """
    #预热模型的信号
    signal_init_model = pyqtSignal(dict)  # 设置触发信号传递的参数数据类型
    #判断模型已经预热后的信号
    signal_init_model_finish = pyqtSignal()

    def __init__(self, arthrosis,bool,classifier_model_path):
        super(InitModelThread, self).__init__()
        self.arthrosis = arthrosis
        self.warm_model = bool
        self.classifier_model_path = classifier_model_path

    # @pysnooper.snoop()
    def run(self):
        #进行模型预热
        if self.warm_model:
            self.signal_init_model_finish.emit()
        else:
            for key,value in self.arthrosis.items():
                if key == 'MCPFifth' or key == 'DIPFifth' or key == 'PIPFifth' or key =='MIPFifth':
                    continue
                model_path = self.classifier_model_path +'/best_' + self.arthrosis[key][0] + '_net.pdparams'
                para_state_dict = paddle.load(model_path)
                self.arthrosis[key][2] .set_dict(para_state_dict)
            self.signal_init_model.emit(self.arthrosis)
            self.warm_model = True



class InferThread(QThread):  # 建立一个任务线程类,  推理任务
    """
    初始化模型
    """
    signal_infer_fail = pyqtSignal() #推理失败的信号
    signal_infer_result = pyqtSignal(dict)  #这信号用来传递推理结果

    def __init__(self, arthrosis,config,im,mode,setting):
        super(InferThread, self).__init__()
        self.config = config
        self.im = im
        self.Hand = True
        self.get_img_mode = mode
        self.arthrosis = arthrosis
        self.finally_results = {}
        self.setting = setting
        self.handle = -1


    # @pysnooper.snoop()
    def run(self):  # 在启动线程后任务从这个函数里面开始执行
        # 预测，分两步，一步是目标检测所有关节，二步是每个关节进行分类
        # 目标检测开始推理，推理的结果保存在results中

        try:
            self.handle = ctypes.windll.kernel32.OpenThread(
                win32con.PROCESS_ALL_ACCESS, False, int(QThread.currentThreadId()))
        except Exception as e:
            print('get thread handle failed', e)
        classifer = {}
        label = {}
        results = self.predict_detection(self.config, self.im)
        for box in results['boxes']:
            # 用一个字典保存box的left,top, right, bottom, box[0],是类别名
            if int(box[0]) not in classifer:
                classifer[int(box[0])] = []
                classifer[int(box[0])].append([int(box[2]), int(box[3]), int(box[4]), int(box[5])])
            else:
                classifer[int(box[0])].append([int(box[2]), int(box[3]), int(box[4]), int(box[5])])
        # 判断预测结果，如果预测错误就报错，终止
        self.predict_error(results, classifer)  # 改成发送推理失败的信号
        # 判断左手还是右手
        self.left_or_right(classifer)
        # 根据left,top, right, bottom  的位置细分关节
        label = self.label_classifier(classifer)
        if self.get_img_mode == 1:
            # 读取图片，根据box的位置切割小关节，用来分类
            image = cv2.imdecode(np.fromfile(self.im, dtype=np.uint8), 0)
        elif self.get_img_mode == 0 or self.get_img_mode == -1:
            image = self.im.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for key, value in label.items():
            # 每个关节通过模型推理，得到对应的等级
            category = self.arthrosis[key]
            left, top, right, bottom = value
            # 从原图根据检测出来的boxes 抠出来，传入分类模型中进行预测
            image_temp = image[top:bottom, left:right]
            # 预测等级
            grade = self.predict_class(image_temp, key)
            self.finally_results[key] = [grade, image_temp.copy()]
        self.signal_infer_result.emit(self.finally_results)


    # @pysnooper.snoop()
    def predict_detection(self,config,im):
        #在手部X图上进行目标检测预测
        detector = Detector(
            config, self.setting['yolo_model_dir'], use_gpu=self.setting['use_gpu'], run_mode=self.setting['run_mode'])
        results = detector.predict(im, self.setting['threshold'])
        return results

    # @pysnooper.snoop()
    def predict_class(self, im,key):
        # 预测小关节的等级
        from paddle.vision.transforms import Compose, Resize, Normalize, Transpose
        transforms = Compose([Resize(size=(224, 224)),
                              Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], data_format='HWC'),
                              Transpose()])
        self.arthrosis[key][2].eval()
        im = np.expand_dims(im, 2)
        infer_data = transforms(im)
        infer_data = np.expand_dims(infer_data, 0)
        infer_data = paddle.to_tensor(infer_data, dtype='float32')
        result = self.arthrosis[key][2](infer_data)[0]  # 关键代码，实现预测功能
        result = np.argmax(result.numpy())  # 获得最大值所在的序号
        return result

    # @pysnooper.snoop()
    def predict_error(self,results,classifer):
        #设置一些检测，假如检测出来不是21个关节，或者是7类关节，但是不是目标关节，就报错
        if len(results['boxes']) != 21:
            self.signal_infer_fail.emit()
            ctypes.windll.kernel32.TerminateThread(self.handle, 0)
        if len(classifer) != 7:
            self.signal_infer_fail.emit()
            ctypes.windll.kernel32.TerminateThread(self.handle, 0)
        if len(classifer) > 3:
            if len(classifer[0]) != 1 or len(classifer[1]) != 1 or len(classifer[2]) != 1:
                self.signal_infer_fail.emit()
                ctypes.windll.kernel32.TerminateThread(self.handle, 0)
            if len(classifer[3]) != 4 or len(classifer[4]) != 5 or len(classifer[5]) != 4 or len(classifer[6]) != 5:
                self.signal_infer_fail.emit()
                ctypes.windll.kernel32.TerminateThread(self.handle, 0)

    # @pysnooper.snoop()
    def label_classifier(self,classifer):
        #根据坐标点细分关节
        label = dict()
        label['Radius'] = classifer[0][0]
        label['Ulna'] = classifer[1][0]
        label['MCPFirst'] = classifer[2][0]

        # 4个MCP中，根据left的大到小排列，分出第三手指掌骨，和第五手指掌骨，因为只需要第三和第五掌骨，其他同理
        MCP = sorted(classifer[3], key=(lambda x: [x[0]]), reverse=self.Hand)
        label['MCPThird'] = MCP[1]
        label['MCPFifth'] = MCP[3]

        # 5个ProximalPhalanx中，根据left的大到小排列，分出第一近节指骨，第三近节指骨，第五近节指骨
        PIP = sorted(classifer[4], key=(lambda x: [x[0]]), reverse=self.Hand)
        label['PIPFirst'] = PIP[0]
        label['PIPThird'] = PIP[2]
        label['PIPFifth'] = PIP[4]

        # 4个MiddlePhalanx中，根据left的大到小排列，分出第三中节指骨，第三中节指骨
        MIP = sorted(classifer[5], key=(lambda x: [x[0]]), reverse=self.Hand)
        label['MIPThird'] = MIP[1]
        label['MIPFifth'] = MIP[3]

        # 5个DistalPhalanx中，根据left的大到小排列，分出第一远节指骨，第三远节指骨，第五远节指骨
        DIP = sorted(classifer[6], key=(lambda x: [x[0]]), reverse=self.Hand)
        label['DIPFirst'] = DIP[0]
        label['DIPThird'] = DIP[2]
        label['DIPFifth'] = DIP[4]

        return label

    # @pysnooper.snoop()
    def left_or_right(self,classifer):
        #判断是左手还是右手
        if classifer[2][0][0] > classifer[1][0][0]:
            self.Hand = True
        else:
            self.Hand = False


class BoneAge(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("AI骨龄计算(未预热)")
        self.cwd = 'C:\\'  # 获取当前程序文件位置
        self.clipboard = QtWidgets.QApplication.clipboard()#创建剪贴板
        self.clipboard.clear()#清空剪贴板

        self.CaptureScreen = CaptureScreen()#实例化截图工具
        self.captureImage = object() #用来保存截图工具返回来的截图

        self.get_img_mode = 1  #获取图片的方式，1是通过打开图片获取，0是通过截图获取, -1通过剪贴板

        self.finally_results = {}#保存最终结果
        self.warm_model = False #判断是否模型是否预热
        self.setting = {} #设置

        self.read_config() #读取设置配置文件

        self.image_file_path = ''  #图片路径

        self.Hand = True # True 是左手， False是右手，True是左手，一般是左手

        self.arthrosis = Arthrosis # 对应13个关节的相关信息，名字，每个关节多少个等级。
        self.init_arthrosis()#初始化13个关节的模型变量

        self.init_ui()#初始化某些UI界面

        #推理区，给按钮绑定槽函数
        self.open_bn.clicked.connect(self.slot_btn_chooseFile)
        self.infer_bn.clicked.connect(self.infer_start)
        self.create_bn.clicked.connect(self.create_report)
        self.screen_bn.clicked.connect(self.screen_start)
        
        for key in Arthrosis.keys():
            #每一个图谱Qlabel都添加一个鼠标点击事件
            eval('self.{}_label'.format(key)).clicked.connect(self.photo_link)
            #每个等级下拉选项都添加一个修改下拉值的事件
            comcoBox = eval('self.{}_cbox'.format(key))
            comcoBox.addItems(list(str(x) for x in range(self.arthrosis[key][1] + 1)))

        #报告区
        self.describe_bn.clicked.connect(self.copy_report)

        #设置区
        self.sure_bn.clicked.connect(self.create_config)
        self.grade_width_hs.valueChanged.connect(lambda:self.value_change('grade_width_hs'))
        self.pre_width_hs.valueChanged.connect(lambda:self.value_change('pre_width_hs'))
        self.label_width_hs.valueChanged.connect(lambda:self.value_change('label_width_hs'))
        self.threshold_hs.valueChanged.connect(lambda :self.value_change('threshold_hs'))
        self.font_size_hs.valueChanged.connect(lambda: self.value_change('font_size_hs'))

        #实例化线程对象
        self.init_model_thread = InitModelThread(self.arthrosis,self.warm_model, self.setting['classifier_model_dir'])
        #绑定预热模型槽函数
        self.init_model_thread.signal_init_model.connect(self.init_model)
        #绑定模型已经预热后的槽函数
        self.init_model_thread.signal_init_model_finish.connect(self.init_model_finish)

        #主程序的菜单栏
        self.actionWarm.triggered.connect(self.start_init_model)
        self.actionOpen.triggered.connect(self.slot_btn_chooseFile)
        self.actionClose.triggered.connect(self.close)
        self.actionWhole.triggered.connect(self.whole_html)
        self.actionRUS_CHN.triggered.connect(self.RUS_CHN)
        self.actionAbout.triggered.connect(self.about)


    def init_ui(self):
        """
        一开始的界面初始化
        :return:
        """
        self.statusBar().hide()#隐藏状态栏
        self.showMaximized()#最大化
        self.is_hide(False)#隐藏一些控件

        self.tabWidget.setCurrentIndex(0)
        self.tabWidget.setTabShape(QTabWidget.Triangular)
        self.tabWidget.setStyleSheet("QTabBar::tab { height: 35px; width:160px;font:12pt '黑体';color: black;}")

        self.setting_init() #初始化设置界面

    def init_arthrosis(self):
        #初始化模型变量
        model_MCPFirst = paddle.vision.models.resnet50(num_classes=self.arthrosis['MCPFirst'][1])
        self.arthrosis['MCPFirst'].append(model_MCPFirst)
        model_MCPThird = paddle.vision.models.resnet50(num_classes=self.arthrosis['MCPThird'][1])
        self.arthrosis['MCPThird'].append(model_MCPThird)
        self.arthrosis['MCPFifth'].append(model_MCPThird)

        model_DIPFirst = paddle.vision.models.resnet50(num_classes=self.arthrosis['DIPFirst'][1])
        self.arthrosis['DIPFirst'].append(model_DIPFirst)
        model_DIPThird = paddle.vision.models.resnet50(num_classes=self.arthrosis['DIPThird'][1])
        self.arthrosis['DIPThird'].append(model_DIPThird)
        self.arthrosis['DIPFifth'].append(model_DIPThird)

        model_PIPFirst = paddle.vision.models.resnet50(num_classes=self.arthrosis['PIPFirst'][1])
        self.arthrosis['PIPFirst'].append(model_PIPFirst)
        model_PIPThird = paddle.vision.models.resnet50(num_classes=self.arthrosis['PIPThird'][1])
        self.arthrosis['PIPThird'].append(model_PIPThird)
        self.arthrosis['PIPFifth'].append(model_PIPThird)

        model_MIPThird = paddle.vision.models.resnet50(num_classes=self.arthrosis['MIPThird'][1])
        self.arthrosis['MIPThird'].append(model_MIPThird)
        self.arthrosis['MIPFifth'].append(model_MIPThird)

        model_Radius = paddle.vision.models.resnet50(num_classes=self.arthrosis['Radius'][1])
        self.arthrosis['Radius'].append(model_Radius)
        model_Ulna = paddle.vision.models.resnet50(num_classes=self.arthrosis['Ulna'][1])
        self.arthrosis['Ulna'].append(model_Ulna)

    def init_parameter(self):
        #初始化一些变量
        self.is_hide(False)  # 每次新推理都隐藏展示区
        paddle.set_device('gpu' if self.setting['use_gpu'] else 'cpu' )
        self.finally_results = {}
        #读取设置文件
        self.read_config()
        #清除报告区的描述
        self.describe_te.clear()

        #设置推理状态信息：
        self.infer_state.setText("推理状态:推理中")

        if self.setting['clipboard_cb']:
            #检查设置是不是选择了剪贴板模式
            self.get_img_mode = -1





    """模型预热相关的线程函数  开始"""
    # @pysnooper.snoop()
    def start_init_model(self):
        self.setWindowTitle("AI骨龄计算(模型正在预热中，请耐性等候,预计需要十秒到一分钟不等)")
        self.init_model_thread.start()

    # @pysnooper.snoop()
    def init_model(self,arthrosis):
        self.warm_model = True
        self.setWindowTitle("AI骨龄计算(已预热)")
        QMessageBox.information(self, "信息", "模型预热成功，可以进行推理！", QMessageBox.Yes, QMessageBox.Yes)
        self.arthrosis = arthrosis
        #预热模型后关闭线程
        self.init_model_thread.quit()

    # @pysnooper.snoop()
    def init_model_finish(self):
        if self.warm_model:
            QMessageBox.information(self, "信息", "模型已经预热成功，无需预热。", QMessageBox.Yes, QMessageBox.Yes)
            self.setWindowTitle("AI骨龄计算(已预热)")
        self.init_model_thread.quit()

    """模型预热相关的线程函数  结束"""

    """推理相关线程函数     开始"""
    # @pysnooper.snoop()
    def infer_start(self):
        self.infer_bn.setEnabled(False)
        if self.warm_model:
            self.init_parameter()
            config = Config(self.setting['yolo_model_dir'])
            if self.get_img_mode == 1:#获取图片的方式，1是通过打开图片获取，0是通过截图获取, -1通过剪贴板
                if self.image_file_path:
                    # 创建推理线程
                    self.infer_thread = InferThread( self.arthrosis,
                                                     config,
                                                     self.image_file_path,
                                                     self.get_img_mode,
                                                     self.setting)
                    # 绑定推理失败的槽函数
                    self.infer_thread.signal_infer_fail.connect(self.infer_fail)
                    # 绑定推理成功的槽函数
                    self.infer_thread.signal_infer_result.connect(self.infer_result)
                    self.infer_thread.start()
                else:
                    QMessageBox.warning(self, "警告", "需要打开一张手部X光图才能推理！", QMessageBox.Yes, QMessageBox.Yes)
                    self.infer_state.setText("推理状态:无")
            elif self.get_img_mode == 0:#截图模式
                if self.captureImage is not None:
                    #创建推理线程
                    self.infer_thread = InferThread( self.arthrosis,
                                                     config,
                                                     self.captureImage,
                                                     self.get_img_mode,
                                                     self.setting)
                    # 绑定推理失败的槽函数
                    self.infer_thread.signal_infer_fail.connect(self.infer_fail)
                    # 绑定推理成功的槽函数
                    self.infer_thread.signal_infer_result.connect(self.infer_result)
                    self.infer_thread.start()
                else:
                    QMessageBox.warning(self, "警告", "没有截到手部X光图片", QMessageBox.Yes, QMessageBox.Yes)
                    self.infer_state.setText("推理状态:无")
            elif self.get_img_mode == -1:#剪贴板模式:
                self.image_path_te.clear()

                mdata = self.clipboard.mimeData()#获取剪贴板内容
                if mdata.hasImage():#判断是否是图像
                    qimage = self.clipboard.image()#获取图像
                    image = self.qimage_to_cvimg(qimage)#转换numpy格式
                    # 创建推理线程
                    self.infer_thread = InferThread( self.arthrosis,
                                                     config,
                                                     image,
                                                     self.get_img_mode,
                                                     self.setting)
                    # 绑定推理失败的槽函数
                    self.infer_thread.signal_infer_fail.connect(self.infer_fail)
                    # 绑定推理成功的槽函数
                    self.infer_thread.signal_infer_result.connect(self.infer_result)
                    self.infer_thread.start()
                else:
                    QMessageBox.warning(self, "警告", "使用剪贴板模式，但是剪贴板中没有图片内容。\n"
                                                    "到‘设置区’可以关闭剪贴板模式。", QMessageBox.Yes, QMessageBox.Yes)
                    self.infer_state.setText("推理状态:无")
                self.clipboard.clear()

        else:
            QMessageBox.warning(self, "警告", "AI未预热，请到菜单【模型】进行预热。\n"
                                            "(只需预热一次，时间需要十几秒到一分钟不等)", QMessageBox.Yes, QMessageBox.Yes)
        self.infer_bn.setEnabled(True)

    # @pysnooper.snoop()
    def infer_fail(self):
        #推理失败的情况
        # self.infer_thread.quit()
        # ret = ctypes.windll.kernel32.TerminateThread(self.infer_thread.handle, 0)
        QMessageBox.warning(self, "警告", "推理失败！", QMessageBox.Yes, QMessageBox.Yes)
        self.infer_state.setText("推理状态:失败")


    # @pysnooper.snoop()
    def infer_result(self, finally_result):
        #推理成功，并显示结果
        self.finally_results = finally_result
        self.show_result()
        QMessageBox.information(self, "信息", "推理完成！", QMessageBox.Yes, QMessageBox.Yes)
        self.infer_thread.quit()
        # 设置推理状态信息：
        self.infer_state.setText("推理状态:推理完成")
    """推理相关线程函数     结束"""


    """推理区截图功能、计算骨龄和显示图片、显示图谱网页等相关的函数  开始"""
    # @pysnooper.snoop()
    def calc_boneAge(self,sex):
        score = 0
        for key, value in self.finally_results.items():
            # 根据每个关节的等级，计算总得分
            score += SCORE[sex][key][value[0]]
        # 计算对应的骨龄
        boneAge = calcBoneAge(score, sex)
        return score, boneAge

    # @pysnooper.snoop()
    def create_report(self):
        #生成报告
        sex = self.sexBx.currentText()
        if self.finally_results:
            for key in self.finally_results.keys():
                #读取每个等级comcobox 的值
                comcoBox = eval('self.{}_cbox'.format(key))
                value  = int(comcoBox.currentText())
                self.finally_results[key][0] = value
            score, boneAge = self.calc_boneAge(sex)
            report = "第一掌骨骺分级{}级，得{}分；第三掌骨骨骺分级{}级，得{}分；第五掌骨骨骺分级{}级，得{}分；\n" \
                     "第一近节骨骺分级{}级，得{}分；第三近节骨骺分级{}级，得{}分；第五近节骨骺分级{}级，得{}分；\n" \
                     "第三中节指骨骨骺分级{}级，得{}分；第五中节指骨骨骺分级{}级，得{}分；\n" \
                     "第一远节指骨骨骺分级{}级，得{}分；第三远节指骨骨骺分级{}级，得{}分；第五远节指骨骨骺分级{}级，得{}分；\n" \
                     "尺骨骨骺分级{}级，得{}分；桡骨骨骺分级{}级，得{}分。\n\n" \
                     "RUS-CHN分级计分法，受检儿CHN总得分：{}分，骨龄约为{}岁。".format(
                self.finally_results['MCPFirst'][0] , SCORE[sex]['MCPFirst'][self.finally_results['MCPFirst'][0]], \
                self.finally_results['MCPThird'][0] , SCORE[sex]['MCPThird'][self.finally_results['MCPThird'][0]], \
                self.finally_results['MCPFifth'][0] , SCORE[sex]['MCPFifth'][self.finally_results['MCPFifth'][0]], \
                self.finally_results['PIPFirst'][0] , SCORE[sex]['PIPFirst'][self.finally_results['PIPFirst'][0]], \
                self.finally_results['PIPThird'][0] , SCORE[sex]['PIPThird'][self.finally_results['PIPThird'][0]], \
                self.finally_results['PIPFifth'][0] , SCORE[sex]['PIPFifth'][self.finally_results['PIPFifth'][0]], \
                self.finally_results['MIPThird'][0] , SCORE[sex]['MIPThird'][self.finally_results['MIPThird'][0]], \
                self.finally_results['MIPFifth'][0] , SCORE[sex]['MIPFifth'][self.finally_results['MIPFifth'][0]], \
                self.finally_results['DIPFirst'][0] , SCORE[sex]['DIPFirst'][self.finally_results['DIPFirst'][0]], \
                self.finally_results['DIPThird'][0] , SCORE[sex]['DIPThird'][self.finally_results['DIPThird'][0]], \
                self.finally_results['DIPFifth'][0] , SCORE[sex]['DIPFifth'][self.finally_results['DIPFifth'][0]], \
                self.finally_results['Ulna'][0] , SCORE[sex]['Ulna'][self.finally_results['Ulna'][0]], \
                self.finally_results['Radius'][0] , SCORE[sex]['Radius'][self.finally_results['Radius'][0]], \
                score, boneAge)

            font = QFont()
            font_size = int(self.setting['font_size'])
            # font.setFamily("Arial")  # 括号里可以设置成自己想要的其它字体
            font.setPointSize(font_size)  # 括号里的数字可以设置成自己想要的字体大小
            self.describe_te.setFont(font)
            self.describe_te.setPlainText(report)
            self.tabWidget.setCurrentIndex(1)
        else:
            QMessageBox.warning(self, "警告", "AI未推理，报告为空！请推理", QMessageBox.Yes , QMessageBox.Yes)

    # @pysnooper.snoop()
    def show_result(self):
        #展示预测结果
        if self.finally_results:
            self.is_hide(True)
            for key, value in self.finally_results.items():
                #展示预测等级写入Qlabel
                grade = eval('self.{}_text'.format(key))
                grade.setMinimumWidth(int(self.setting['grade_line_width']))
                grade.setText('{}:{}'.format(key, value[0]))

                # 展示预测等级写入Qcombox  给医生可以选择修改
                comcoBox = eval('self.{}_cbox'.format(key))
                comcoBox.setCurrentIndex(int(value[0]))

                #展示预测的关节图片
                pre = eval('self.{}_pre'.format(key))
                qimage = self.cv2_to_qimage(value[1],int(self.setting['pre_line_width']),key)
                # pre.resize(QtCore.QSize(self.img_rgb.shape[1], self.img_rgb.shape[0]))
                pre.setPixmap(QPixmap.fromImage(qimage))

                #展示预测的等级对应标准图谱
                label = eval('self.{}_label'.format(key))
                label_path = 'image/atlas/' + str(key) + '/' + str(value[0]) +'.png'
                image = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), 0)
                qimage = self.cv2_to_qimage(image,int(self.setting['label_line_width']))
                label.setPixmap(QPixmap.fromImage(qimage))
        else:
            QMessageBox.warning(self, "警告", "推理失败！", QMessageBox.Yes, QMessageBox.Yes)

    # @pysnooper.snoop()
    def photo_link(self):
        #图谱展示区 点击后打开网页展示对应的等级的图谱
        try:
            html_name = 'boneagehtml/' + str(self.sender().objectName()).split('_')[0] + '.html'
            web.open('file:/' + os.path.realpath(html_name),new=0,autoraise=True)
        except Exception as e:
            QMessageBox.warning(self, "警告", 'Qlabel.clicked :' + e, QMessageBox.Yes, QMessageBox.Yes)

    # @pysnooper.snoop()
    def cv2_to_qimage(self,img,new_width,classifier =None):
        height ,width =img.shape[0] ,img.shape[1]
        #有些关节长宽比例太大，不好看，进行裁剪
        if classifier  == 'MCPThird' or classifier == 'MCPFifth':
            height = int(height/5*3)
            img = img[:height,:]
        elif classifier  == 'Ulna' or classifier == 'Radius':
            height = int(height / 5 * 3)
            img = img[:height, :]
        elif classifier  == 'PIPThird' or classifier == 'PIPFifth':
            height = int(height / 5 * 2)
            img = img[height::, :]
        elif classifier  == 'DIPFirst' or classifier == 'DIPThird' or classifier == 'DIPFifth':
            height = int(height / 3)
            img = img[height::, :]

        pacent = round(float(new_width / float(width)), 2)
        img = cv2.resize(img, None, fx=pacent, fy=pacent, interpolation=cv2.INTER_AREA)
        qimage = QImage(img[:], img.shape[1], img.shape[0], img.shape[1] * 1,
                        QImage.Format_Grayscale8)
        return qimage

    def is_hide(self,switch):
        #一开始运行就隐藏所有展示区的控件
        for key in Arthrosis.keys():
            eval('self.{}_text'.format(key)).setVisible(switch)
            eval('self.{}_cbox'.format(key)).setVisible(switch)
            eval('self.{}_pre'.format(key)).setVisible(switch)
            eval('self.{}_label'.format(key)).setVisible(switch)
        self.sexBx.setVisible(switch)
        self.create_bn.setVisible(switch)

    # @pysnooper.snoop()
    def slot_btn_chooseFile(self):
        #选择文件
        fileName_choose, filetype = QFileDialog.getOpenFileName(self,'打开图片',self.cwd,
                                                                'Image files (*.jpg *.gif *.png *.jpeg)')
        if fileName_choose == "":
            self.image_file_path = ''
            if self.setting['clipboard_cb']:
                # 检查设置是不是选择了剪贴板模式
                self.get_img_mode = -1
        else:
            self.cwd = os.path.dirname(fileName_choose)
            self.image_file_path = fileName_choose
            self.image_path_te.setText(fileName_choose)
            self.get_img_mode = 1

    # @pysnooper.snoop()
    def screen_start(self):
        #开始截图，运行截图工具，截图工具的signal信号连接主界面的get_screen槽函数
        self.showMinimized()  # 最小化主窗口
        time.sleep(0.3)
        self.CaptureScreen.captureFullScreen()
        # self.CaptureScreen.showMaximized()
        self.CaptureScreen.show()
        self.CaptureScreen.signal.connect(self.get_screen)

    # @pysnooper.snoop()
    def get_screen(self,img):
        #得到截图
        self.showMaximized()#最大化主窗口
        #从截图工具类 获取返回的截图
        if img:
            self.captureImage = self.qtpixmap_to_cvimg(img)
            self.get_img_mode = 0 #设置截图模式
            self.image_path_te.clear()
        else:
            self.captureImage = None
            if self.setting['clipboard_cb']:
                # 检查设置是不是选择了剪贴板模式
                self.get_img_mode = -1

    def qtpixmap_to_cvimg(self,qtpixmap):
        #QTpixmap  图像转 opencv格式图像
        qimg = qtpixmap.toImage()
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]
        return result

    def qimage_to_cvimg(self,qimg):
        temp_shape = (qimg.height(), qimg.bytesPerLine() * 8 // qimg.depth())
        temp_shape += (4,)
        ptr = qimg.bits()
        ptr.setsize(qimg.byteCount())
        result = np.array(ptr, dtype=np.uint8).reshape(temp_shape)
        result = result[..., :3]
        return result

    """推理区截图功能、计算骨龄和显示图片、显示图谱网页等相关的函数  结束"""


    """报告区 槽函数定义  开始"""
    # @pysnooper.snoop()
    def copy_report(self):
        report = self.describe_te.toPlainText()
        if report :
            pyperclip.copy(report)
        else:
            QMessageBox.warning(self, "警告", "报告为空无法复制", QMessageBox.Yes, QMessageBox.Yes)
    """报告区 槽函数定义  结束"""


    """设置区 槽函数定义  开始"""
    # @pysnooper.snoop()
    def create_config(self,content = None):
        #保存、写入配置文件config.json
        if content :
            with open('config/config.json', 'w', encoding='utf-8') as f:
                content = json.dumps(content, ensure_ascii=False, indent=4, separators=(',', ':'))
                f.write(content)
        else:
            setting = dict()
            setting['threshold'] = round(float(self.threshold_hs.value()) /10.0, 1)
            if self.nogpu_rb.isChecked():
                setting['use_gpu'] = False
            if self.gpu_rb.isChecked():
                setting['use_gpu'] = True
            setting['yolo_model_dir'] = './inference_model/yolov3_darknet_voc'
            setting['classifier_model_dir'] = './inference_model/bone_classifier'

            setting['grade_line_width'] = int(self.grade_width_hs.value())#等级lineEdit控件的宽度
            setting['pre_line_width'] = int(self.pre_width_hs.value())#预测后分割小关节的图片的宽度
            setting['label_line_width'] = int(self.label_width_hs.value())#对应图谱的图片的宽度

            setting['run_mode'] = 'fluid'

            setting['font_size'] = int(self.font_size_hs.value()) #报告显示字体大小

            setting['clipboard_cb'] = self.clipboard_cb.isChecked()  #是否勾选了剪贴板功能

            with open('config/config.json', 'w',encoding='utf-8') as f:
                content = json.dumps(setting,ensure_ascii=False, indent=4, separators=(',', ':'))
                f.write(content)

    def read_config(self):
        #读取配置文件
        if os.path.exists('config/config.json'):
            self.setting = {}
            with open('config/config.json', 'r', encoding='utf-8') as f:
                self.setting = json.load(f)
        else:
            QMessageBox.warning(self, "警告", "配置文件不存在！", QMessageBox.Yes, QMessageBox.Yes)

    def setting_init(self):
        #根据配置文件初始化 设置界面
        self.read_config()

        self.threshold_hs.setValue(self.setting['threshold'] *10)
        self.grade_width_hs.setValue(self.setting['grade_line_width'])
        self.pre_width_hs.setValue(self.setting['pre_line_width'])
        self.label_width_hs.setValue(self.setting['label_line_width'])
        self.font_size_hs.setValue(self.setting['font_size'])

        self.threshold_value.setText(str(self.setting['threshold']))
        self.grade_line_width.setText(str(self.setting['grade_line_width']))
        self.pre_line_width.setText(str(self.setting['pre_line_width']))
        self.label_line_width.setText(str(self.setting['label_line_width']))
        self.font_size_lable.setText(str(self.setting['font_size']))

        self.clipboard_cb.setChecked(self.setting['clipboard_cb'])


        #判断电脑是否 GPU
        device = str(paddle.get_device())
        if 'gpu' in device:
            self.gpu_rb.setChecked(True)
            self.nogpu_rb.setChecked(False)
            self.setting['use_gpu'] = True
        elif 'cpu' in device:
            self.setting['use_gpu'] = False
            self.nogpu_rb.setChecked(True)
            self.gpu_rb.setEnabled(False)
        self.create_config(self.setting)

    # @pysnooper.snoop()
    def value_change(self,object_name):
        #当滑动条改变，改变数值
        if object_name is 'grade_width_hs':
            self.grade_line_width.setText(str(self.grade_width_hs.value()))
        elif object_name is 'pre_width_hs':
            self.pre_line_width.setText(str(self.pre_width_hs.value()))
        elif object_name is 'label_width_hs':
            self.label_line_width.setText(str(self.label_width_hs.value()))
        elif object_name is 'threshold_hs':
            self.threshold_value.setText(str(round(float(self.threshold_hs.value()) /10.0, 1)))
        elif object_name is 'font_size_hs':
            self.font_size_lable.setText(str(self.font_size_hs.value()))
    """设置区 槽函数定义  结束"""



    """菜单栏 槽函数定义    开始"""
    def whole_html(self):
        #菜单"骨龄标准整片图谱"点击后显示的网页
        html_name = 'html/whole.html'
        web.open('file:/' + os.path.realpath(html_name), new=0, autoraise=True)

    def RUS_CHN(self):
        # 菜单"RUS_CHN图谱"点击后显示的网页
        html_name = 'boneagehtml/Radius.html'
        web.open('file:/' + os.path.realpath(html_name), new=0, autoraise=True)

    def about(self):
        # 菜单"关于此项目"点击后显示的网页
        html_name = 'html/about.html'
        web.open('file:/' + os.path.realpath(html_name), new=0, autoraise=True)
    """菜单栏 槽函数定义    结束"""

    def closeEvent(self, event):
        reply = QMessageBox.question(self, '确认框', '确认退出吗？',
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = BoneAge()
    main.show()
    sys.exit(app.exec_())