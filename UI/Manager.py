# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'manager.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_ManageWindow(object):
    def setupUi(self, ManageWindow):
        ManageWindow.setObjectName("ManageWindow")
        ManageWindow.resize(945, 736)
        self.centralwidget = QtWidgets.QWidget(ManageWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(90, 50, 731, 511))
        self.frame.setMinimumSize(QtCore.QSize(731, 511))
        self.frame.setStyleSheet("")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setStyleSheet("")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_4 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(15)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_4.sizePolicy().hasHeightForWidth())
        self.frame_4.setSizePolicy(sizePolicy)
        self.frame_4.setStyleSheet("#frame_4 {\n"
"    background-color: rgba(255,255,255); \n"
"}")
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_4)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setSpacing(0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton = QtWidgets.QPushButton(self.frame_4)
        self.pushButton.setStyleSheet("QPushButton:hover {\n"
"    padding-bottom: 5px; /* 鼠标悬停时按钮底部内边距为5px */\n"
"}\n"
"\n"
"QPushButton {\n"
"    border: none; /* 移除按钮边框 */\n"
"}")
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.horizontalLayout.addWidget(self.frame_4)
        self.frame_5 = QtWidgets.QFrame(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_5.sizePolicy().hasHeightForWidth())
        self.frame_5.setSizePolicy(sizePolicy)
        self.frame_5.setStyleSheet("QPushButton:hover{\n"
"padding-bottom:5px;\n"
"}\n"
"#frame_5 {\n"
"    background-color: rgba(255,255,255); \n"
"}")
        self.frame_5.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.frame_5)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setSpacing(0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_Minimize = QtWidgets.QPushButton(self.frame_5)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("C:/Users/26306/.designer/icon/花花.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Minimize.setIcon(icon)
        self.pushButton_Minimize.setObjectName("pushButton_Minimize")
        self.horizontalLayout_3.addWidget(self.pushButton_Minimize)
        self.pushButton_Close2 = QtWidgets.QPushButton(self.frame_5)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("C:/Users/26306/.designer/icon/郁金香.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_Close2.setIcon(icon1)
        self.pushButton_Close2.setObjectName("pushButton_Close2")
        self.horizontalLayout_3.addWidget(self.pushButton_Close2)
        self.horizontalLayout.addWidget(self.frame_5)
        self.verticalLayout.addWidget(self.frame_2)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(10)
        sizePolicy.setHeightForWidth(self.frame_3.sizePolicy().hasHeightForWidth())
        self.frame_3.setSizePolicy(sizePolicy)
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.frame_3)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(4)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_6.sizePolicy().hasHeightForWidth())
        self.frame_6.setSizePolicy(sizePolicy)
        self.frame_6.setStyleSheet("#frame_6 {\n"
"    background-color: rgba(255, 182, 193); /* 半透明的淡粉色背景 */\n"
"}\n"
"\n"
"QPushButton {\n"
"    border: none; /* 无边框 */\n"
"    color: rgb(255, 255, 255); /* 文字颜色为白色 */\n"
"    font: 14pt \"隶书\"; /* 修正字体大小和字体族的声明 */\n"
"}\n"
"\n"
"QPushButton:hover {\n"
"    color: rgb(189, 189, 189); /* 鼠标悬停时的文字颜色 */\n"
"}\n"
"QPushButton:pressed {\n"
"    color: rgb(100, 100, 100); /* 按钮被按下时的文字颜色 */\n"
"}")
        self.frame_6.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.pushButton_Admin = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_Admin.setObjectName("pushButton_Admin")
        self.verticalLayout_2.addWidget(self.pushButton_Admin)
        self.pushButton_Experiments = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_Experiments.setObjectName("pushButton_Experiments")
        self.verticalLayout_2.addWidget(self.pushButton_Experiments)
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.frame_7 = QtWidgets.QFrame(self.frame_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(15)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_7.sizePolicy().hasHeightForWidth())
        self.frame_7.setSizePolicy(sizePolicy)
        self.frame_7.setStyleSheet("#frame_7 {\n"
"    background-color: rgba(255,255,255); \n"
"}")
        self.frame_7.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_7.setObjectName("frame_7")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame_7)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.stackedWidget = QtWidgets.QStackedWidget(self.frame_7)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.tableWidget = QtWidgets.QTableWidget(self.page)
        self.tableWidget.setGeometry(QtCore.QRect(0, 91, 571, 231))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(5)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(4, item)
        self.pushButton_look = QtWidgets.QPushButton(self.page)
        self.pushButton_look.setGeometry(QtCore.QRect(30, 370, 93, 28))
        self.pushButton_look.setObjectName("pushButton_look")
        self.pushButton_modify = QtWidgets.QPushButton(self.page)
        self.pushButton_modify.setGeometry(QtCore.QRect(210, 370, 93, 28))
        self.pushButton_modify.setObjectName("pushButton_modify")
        self.pushButton_delect = QtWidgets.QPushButton(self.page)
        self.pushButton_delect.setGeometry(QtCore.QRect(370, 370, 93, 28))
        self.pushButton_delect.setObjectName("pushButton_delect")
        self.stackedWidget.addWidget(self.page)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_Email = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_Email.setObjectName("lineEdit_Email")
        self.gridLayout.addWidget(self.lineEdit_Email, 2, 3, 1, 1)
        self.lineEdit_password = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_password.setObjectName("lineEdit_password")
        self.gridLayout.addWidget(self.lineEdit_password, 2, 1, 1, 1)
        self.lineEdit_gender = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_gender.setObjectName("lineEdit_gender")
        self.gridLayout.addWidget(self.lineEdit_gender, 3, 1, 1, 1)
        self.lineEdit_name = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_name.setObjectName("lineEdit_name")
        self.gridLayout.addWidget(self.lineEdit_name, 0, 3, 1, 1)
        self.pushButton_look_3 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_look_3.setObjectName("pushButton_look_3")
        self.gridLayout.addWidget(self.pushButton_look_3, 4, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.page_3)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 2, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.page_3)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 3, 0, 1, 1)
        self.lineEdit_date = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_date.setObjectName("lineEdit_date")
        self.gridLayout.addWidget(self.lineEdit_date, 3, 3, 1, 1)
        self.lineEdit_ID = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_ID.setObjectName("lineEdit_ID")
        self.gridLayout.addWidget(self.lineEdit_ID, 0, 1, 1, 1)
        self.pushButton_modify_3 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_modify_3.setObjectName("pushButton_modify_3")
        self.gridLayout.addWidget(self.pushButton_modify_3, 4, 3, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.page_3)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 0, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.page_3)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.page_3)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 2, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.page_3)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 3, 2, 1, 1)
        self.stackedWidget.addWidget(self.page_3)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.page_2)
        self.tableWidget_2.setGeometry(QtCore.QRect(0, 80, 571, 241))
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(6)
        self.tableWidget_2.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(5, item)
        self.pushButton_look_2 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_look_2.setGeometry(QtCore.QRect(30, 370, 93, 28))
        self.pushButton_look_2.setObjectName("pushButton_look_2")
        self.pushButton_modify_2 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_modify_2.setGeometry(QtCore.QRect(200, 370, 93, 28))
        self.pushButton_modify_2.setObjectName("pushButton_modify_2")
        self.pushButton_delect_2 = QtWidgets.QPushButton(self.page_2)
        self.pushButton_delect_2.setGeometry(QtCore.QRect(400, 370, 93, 28))
        self.pushButton_delect_2.setObjectName("pushButton_delect_2")
        self.stackedWidget.addWidget(self.page_2)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.page_4)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.pushButton_look_4 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_look_4.setObjectName("pushButton_look_4")
        self.gridLayout_2.addWidget(self.pushButton_look_4, 6, 1, 1, 1)
        self.lineEdit_model = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_model.setObjectName("lineEdit_model")
        self.gridLayout_2.addWidget(self.lineEdit_model, 2, 1, 1, 1)
        self.lineEdit_experiment_ID = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_experiment_ID.setObjectName("lineEdit_experiment_ID")
        self.gridLayout_2.addWidget(self.lineEdit_experiment_ID, 1, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.page_4)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 5, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.page_4)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 5, 2, 1, 1)
        self.lineEdit_experiment_result = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_experiment_result.setObjectName("lineEdit_experiment_result")
        self.gridLayout_2.addWidget(self.lineEdit_experiment_result, 5, 3, 1, 1)
        self.pushButton_modify_4 = QtWidgets.QPushButton(self.page_4)
        self.pushButton_modify_4.setObjectName("pushButton_modify_4")
        self.gridLayout_2.addWidget(self.pushButton_modify_4, 6, 3, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.page_4)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 2, 0, 1, 1)
        self.lineEdit_user_ID = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_user_ID.setObjectName("lineEdit_user_ID")
        self.gridLayout_2.addWidget(self.lineEdit_user_ID, 1, 3, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.page_4)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 1, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.page_4)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 2, 2, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.page_4)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 1, 2, 1, 1)
        self.lineEdit_experiment_date = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_experiment_date.setObjectName("lineEdit_experiment_date")
        self.gridLayout_2.addWidget(self.lineEdit_experiment_date, 2, 3, 1, 1)
        self.lineEdit_experiment_image = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_experiment_image.setObjectName("lineEdit_experiment_image")
        self.gridLayout_2.addWidget(self.lineEdit_experiment_image, 5, 1, 1, 1)
        self.stackedWidget.addWidget(self.page_4)
        self.verticalLayout_3.addWidget(self.stackedWidget)
        self.horizontalLayout_4.addWidget(self.frame_7)
        self.verticalLayout.addWidget(self.frame_3)
        ManageWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(ManageWindow)
        self.stackedWidget.setCurrentIndex(2)
        self.pushButton_Close2.clicked.connect(ManageWindow.close) # type: ignore
        self.pushButton_Minimize.clicked.connect(ManageWindow.hide) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(ManageWindow)

    def retranslateUi(self, ManageWindow):
        _translate = QtCore.QCoreApplication.translate
        ManageWindow.setWindowTitle(_translate("ManageWindow", "MainWindow"))
        self.pushButton.setText(_translate("ManageWindow", "机器学习实验管理平台"))
        self.pushButton_Minimize.setText(_translate("ManageWindow", "缩小"))
        self.pushButton_Close2.setText(_translate("ManageWindow", "关闭"))
        self.pushButton_Admin.setText(_translate("ManageWindow", "个人账号管理"))
        self.pushButton_Experiments.setText(_translate("ManageWindow", "实验管理"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("ManageWindow", "ID"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("ManageWindow", "用户名"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("ManageWindow", "密码"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("ManageWindow", "邮箱"))
        item = self.tableWidget.horizontalHeaderItem(4)
        item.setText(_translate("ManageWindow", "注册日期"))
        self.pushButton_look.setText(_translate("ManageWindow", "查看"))
        self.pushButton_modify.setText(_translate("ManageWindow", "修改"))
        self.pushButton_delect.setText(_translate("ManageWindow", "删除"))
        self.pushButton_look_3.setText(_translate("ManageWindow", "查看"))
        self.label_2.setText(_translate("ManageWindow", " 密码："))
        self.label_3.setText(_translate("ManageWindow", " 性别："))
        self.pushButton_modify_3.setText(_translate("ManageWindow", "修改"))
        self.label_4.setText(_translate("ManageWindow", "    用户名："))
        self.label.setText(_translate("ManageWindow", "  ID:"))
        self.label_5.setText(_translate("ManageWindow", "      邮箱："))
        self.label_6.setText(_translate("ManageWindow", "  注册时间："))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("ManageWindow", "用户ID"))
        item = self.tableWidget_2.horizontalHeaderItem(1)
        item.setText(_translate("ManageWindow", "实验ID"))
        item = self.tableWidget_2.horizontalHeaderItem(2)
        item.setText(_translate("ManageWindow", "模型名"))
        item = self.tableWidget_2.horizontalHeaderItem(3)
        item.setText(_translate("ManageWindow", "预测数据路径"))
        item = self.tableWidget_2.horizontalHeaderItem(4)
        item.setText(_translate("ManageWindow", "实验结果"))
        item = self.tableWidget_2.horizontalHeaderItem(5)
        item.setText(_translate("ManageWindow", "实验时间"))
        self.pushButton_look_2.setText(_translate("ManageWindow", "查看"))
        self.pushButton_modify_2.setText(_translate("ManageWindow", "修改"))
        self.pushButton_delect_2.setText(_translate("ManageWindow", "删除"))
        self.pushButton_look_4.setText(_translate("ManageWindow", "查看"))
        self.label_11.setText(_translate("ManageWindow", " 预测数据："))
        self.label_12.setText(_translate("ManageWindow", " 实验结果："))
        self.pushButton_modify_4.setText(_translate("ManageWindow", "修改"))
        self.label_9.setText(_translate("ManageWindow", "  模型名："))
        self.label_7.setText(_translate("ManageWindow", "  实验ID："))
        self.label_10.setText(_translate("ManageWindow", " 实验时间："))
        self.label_8.setText(_translate("ManageWindow", "  用户ID："))
