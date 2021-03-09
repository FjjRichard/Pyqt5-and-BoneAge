import os
import yaml
import numpy as np
import paddle.fluid as fluid
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, qAbs, QRect,pyqtSignal
from PyQt5.QtGui import QPen, QPainter, QColor, QGuiApplication
from preprocess import preprocess, Resize, Normalize, Permute, PadStride
import math
from PyQt5 import QtCore, QtWidgets

"""重写了Qlabel控件，增加鼠标单击事件"""
class MyQLabel(QtWidgets.QLabel):
    # 自定义单击信号
    clicked = QtCore.pyqtSignal()
    # 自定义双击信号
    DoubleClicked = QtCore.pyqtSignal()

    def __int__(self):
        super().__init__()

    # 重写鼠标单击事件
    def mousePressEvent(self, QMouseEvent):  # 单击
        self.clicked.emit()

    # 重写鼠标双击事件
    def mouseDoubleClickEvent(self, e):  # 双击
        self.DoubleClicked.emit()

QtWidgets.QLabel = MyQLabel




"""截图功能 类"""
class CaptureScreen(QWidget):
    # 初始化变量
    beginPosition = None
    endPosition = None
    fullScreenImage = None
    captureImage = None
    isMousePressLeft = None
    painter = QPainter()
    signal = pyqtSignal(object)  # 子界面类创建信号用来绑定主界面类的函数方法

    def __init__(self):
        super(QWidget, self).__init__()
        self.setWindowTitle("截图小工具")
        self.initWindow()   # 初始化窗口
        # self.captureFullScreen()    # 获取全屏

    def initWindow(self):
        self.setMouseTracking(True)     # 鼠标追踪
        self.setCursor(Qt.CrossCursor)  # 设置光标
        self.setWindowFlag(Qt.FramelessWindowHint)  # 窗口无边框
        self.setWindowState(Qt.WindowFullScreen)    # 窗口全屏

    def captureFullScreen(self):
        self.fullScreenImage = QGuiApplication.primaryScreen().grabWindow(QApplication.desktop().winId())
        self.showMaximized()  # 最大化
        self.setWindowFlags(Qt.WindowStaysOnTopHint) #顶置窗口

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.beginPosition = event.pos()
            self.isMousePressLeft = True
        if event.button() == Qt.RightButton:
            # 如果选取了图片,则按一次右键开始重新截图
            if self.captureImage is not None:
                self.captureImage = None
                self.paintBackgroundImage()
                self.update()
            else:
                self.close()

    def mouseMoveEvent(self, event):
        if self.isMousePressLeft is True:
            self.endPosition = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        self.endPosition = event.pos()
        self.isMousePressLeft = False

    def mouseDoubleClickEvent(self, event):
        if self.captureImage is not None:
            self.signal.emit(self.captureImage)
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.signal.emit(None)
            self.close()
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            if self.captureImage is not None:
                # self.saveImage()
                self.signal.emit(self.captureImage)
                self.close()


    def paintBackgroundImage(self):
        shadowColor = QColor(0, 0, 0, 100)  # 黑色半透明
        self.painter.drawPixmap(0, 0, self.fullScreenImage)
        self.painter.fillRect(self.fullScreenImage.rect(), shadowColor)     # 填充矩形阴影

    def paintEvent(self, event):
        self.painter.begin(self)    # 开始重绘
        self.paintBackgroundImage()
        penColor = QColor(30, 144, 245)     # 画笔颜色
        self.painter.setPen(QPen(penColor, 1, Qt.SolidLine, Qt.RoundCap))    # 设置画笔,蓝色,1px大小,实线,圆形笔帽
        if self.isMousePressLeft is True:
            pickRect = self.getRectangle(self.beginPosition, self.endPosition)   # 获得要截图的矩形框
            self.captureImage = self.fullScreenImage.copy(pickRect)         # 捕获截图矩形框内的图片
            self.painter.drawPixmap(pickRect.topLeft(), self.captureImage)  # 填充截图的图片
            self.painter.drawRect(pickRect)     # 画矩形边框
        self.painter.end()  # 结束重绘

    def getRectangle(self, beginPoint, endPoint):
        pickRectWidth = int(qAbs(beginPoint.x() - endPoint.x()))
        pickRectHeight = int(qAbs(beginPoint.y() - endPoint.y()))
        pickRectTop = beginPoint.x() if beginPoint.x() < endPoint.x() else endPoint.x()
        pickRectLeft = beginPoint.y() if beginPoint.y() < endPoint.y() else endPoint.y()
        pickRect = QRect(pickRectTop, pickRectLeft, pickRectWidth, pickRectHeight)
        # 避免高度宽度为0时候报错
        if pickRectWidth == 0:
            pickRect.setWidth(2)
        if pickRectHeight == 0:
            pickRect.setHeight(2)
        return pickRect

    def saveImage(self):
        self.captureImage.save('picture.png', quality=100)   # 保存图片到当前文件夹中


"""根据总分计算对应的年龄"""
def calcBoneAge(score, sex):
    if sex == 'boy':
        boneAge = 2.01790023656577 + (-0.0931820870747269)*score + math.pow(score,2)*0.00334709095418796 +\
        math.pow(score,3)*(-3.32988302362153E-05) + math.pow(score,4)*(1.75712910819776E-07) +\
        math.pow(score,5)*(-5.59998691223273E-10) + math.pow(score,6)*(1.1296711294933E-12) +\
        math.pow(score,7)* (-1.45218037113138e-15) +math.pow(score,8)* (1.15333377080353e-18) +\
        math.pow(score,9)*(-5.15887481551927e-22) +math.pow(score,10)* (9.94098428102335e-26)
        return round(boneAge,2)
    elif sex == 'girl':
        boneAge = 5.81191794824917 + (-0.271546561737745)*score + \
        math.pow(score,2)*0.00526301486340724 + math.pow(score,3)*(-4.37797717401925E-05) +\
        math.pow(score,4)*(2.0858722025667E-07) +math.pow(score,5)*(-6.21879866563429E-10) + \
        math.pow(score,6)*(1.19909931745368E-12) +math.pow(score,7)* (-1.49462900826936E-15) +\
        math.pow(score,8)* (1.162435538672E-18) +math.pow(score,9)*(-5.12713017846218E-22) +\
        math.pow(score,10)* (9.78989966891478E-26)
        return round(boneAge,2)


"""目标检测推理 类"""
class Detector(object):
    def __init__(self,
                 config,
                 model_dir,
                 use_gpu=False,
                 run_mode='fluid',
                 threshold=0.5):
        self.config = config
        self.predictor = self.load_predictor(
            model_dir,
            run_mode=run_mode,
            min_subgraph_size=self.config.min_subgraph_size,
            use_gpu=use_gpu)

    # @pysnooper.snoop()
    def preprocess(self, im):
        preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            if op_type == 'Resize':
                new_op_info['arch'] = self.config.arch
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        inputs = self.create_inputs(im, im_info, self.config.arch)
        return inputs, im_info

    # @pysnooper.snoop()
    def postprocess(self, np_boxes, np_masks, np_lmk, im_info, threshold=0.5):
        results = {}
        expect_boxes = (np_boxes[:, 1] > threshold) & (np_boxes[:, 0] > -1)
        np_boxes = np_boxes[expect_boxes, :]
        results['boxes'] = np_boxes
        return results

    # @pysnooper.snoop()
    def predict(self,
                image,
                threshold=0.2,
                warmup=0,
                repeats=1,
                run_benchmark=False):

        inputs, im_info = self.preprocess(image)
        np_boxes, np_masks, np_lmk = None, None, None

        input_names = self.predictor.get_input_names()
        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_tensor(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        for i in range(repeats):
            self.predictor.zero_copy_run()
            output_names = self.predictor.get_output_names()
            boxes_tensor = self.predictor.get_output_tensor(output_names[0])
            np_boxes = boxes_tensor.copy_to_cpu()

        results = []
        if not run_benchmark:
            results = self.postprocess(
                np_boxes, np_masks, np_lmk, im_info, threshold=threshold)
        return results

    # @pysnooper.snoop()
    def create_inputs(self,im, im_info, model_arch='YOLO'):
        inputs = {}
        inputs['image'] = im
        origin_shape = list(im_info['origin_shape'])
        resize_shape = list(im_info['resize_shape'])
        pad_shape = list(im_info['pad_shape']) if im_info[
                                                      'pad_shape'] is not None else list(im_info['resize_shape'])
        scale_x, scale_y = im_info['scale']
        im_size = np.array([origin_shape]).astype('int32')
        inputs['im_size'] = im_size
        return inputs

    # @pysnooper.snoop()
    def load_predictor(self,model_dir,
                       run_mode='fluid',
                       batch_size=1,
                       use_gpu=False,
                       min_subgraph_size=3):

        if not use_gpu and not run_mode == 'fluid':
            raise ValueError(
                "Predict by TensorRT mode: {}, expect use_gpu==True, but use_gpu == {}"
                    .format(run_mode, use_gpu))
        if run_mode == 'trt_int8':
            raise ValueError("TensorRT int8 mode is not supported now, "
                             "please use trt_fp32 or trt_fp16 instead.")
        precision_map = {
            'trt_int8': fluid.core.AnalysisConfig.Precision.Int8,
            'trt_fp32': fluid.core.AnalysisConfig.Precision.Float32,
            'trt_fp16': fluid.core.AnalysisConfig.Precision.Half
        }
        config = fluid.core.AnalysisConfig(
            os.path.join(model_dir, '__model__'),
            os.path.join(model_dir, '__params__'))
        if use_gpu:
            config.enable_use_gpu(100, 0)
            config.switch_ir_optim(True)
        else:
            config.disable_gpu()

        if run_mode in precision_map.keys():
            config.enable_tensorrt_engine(
                workspace_size=1 << 10,
                max_batch_size=batch_size,
                min_subgraph_size=min_subgraph_size,
                precision_mode=precision_map[run_mode],
                use_static=False,
                use_calib_mode=False)

        config.disable_glog_info()
        config.enable_memory_optim()
        config.switch_use_feed_fetch_ops(False)
        predictor = fluid.core.create_paddle_predictor(config)
        return predictor


"""目标检测配置 类"""
class Config():
    def __init__(self, model_dir):
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)
        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']