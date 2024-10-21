# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Interface.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 20, 731, 511))
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
        self.pushButton_Home = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_Home.setObjectName("pushButton_Home")
        self.verticalLayout_2.addWidget(self.pushButton_Home)
        self.pushButton_Model = QtWidgets.QPushButton(self.frame_6)
        self.pushButton_Model.setObjectName("pushButton_Model")
        self.verticalLayout_2.addWidget(self.pushButton_Model)
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
        self.stackedWidget.setStyleSheet("")
        self.stackedWidget.setObjectName("stackedWidget")
        self.page_Home = QtWidgets.QWidget()
        self.page_Home.setObjectName("page_Home")
        self.pushButton_Experiment_Info = QtWidgets.QPushButton(self.page_Home)
        self.pushButton_Experiment_Info.setGeometry(QtCore.QRect(420, 90, 136, 36))
        self.pushButton_Experiment_Info.setStyleSheet("QPushButton {  \n"
"                background-color: #e6e6fa; /* 浅紫色背景 */  \n"
"                color: #ffffff; /* 白色字体 */  \n"
"                border: none; /* 移除边框 */  \n"
"                padding: 10px 20px; /* 内边距 */  \n"
"                text-align: center; /* 文本居中 */  \n"
"                font-size: 16px; /* 字体大小 */  \n"
"                cursor: pointer; /* 鼠标悬停时显示指针 */  \n"
"            }  \n"
"            QPushButton:hover {  \n"
"                background-color: #d1d1f0; /* 鼠标悬停时的背景色 */  \n"
"            }  \n"
"            QPushButton:pressed {  \n"
"                background-color: #b3b3e6; /* 按钮按下时的背景色 */  \n"
"            }  ")
        self.pushButton_Experiment_Info.setObjectName("pushButton_Experiment_Info")
        self.pushButton_Personal_Info = QtWidgets.QPushButton(self.page_Home)
        self.pushButton_Personal_Info.setGeometry(QtCore.QRect(420, 210, 136, 36))
        self.pushButton_Personal_Info.setStyleSheet("QPushButton {  \n"
"                background-color: #e6e6fa; /* 浅紫色背景 */  \n"
"                color: #ffffff; /* 白色字体 */  \n"
"                border: none; /* 移除边框 */  \n"
"                padding: 10px 20px; /* 内边距 */  \n"
"                text-align: center; /* 文本居中 */  \n"
"                font-size: 16px; /* 字体大小 */  \n"
"                cursor: pointer; /* 鼠标悬停时显示指针 */  \n"
"            }  \n"
"            QPushButton:hover {  \n"
"                background-color: #d1d1f0; /* 鼠标悬停时的背景色 */  \n"
"            }  \n"
"            QPushButton:pressed {  \n"
"                background-color: #b3b3e6; /* 按钮按下时的背景色 */  \n"
"            }  ")
        self.pushButton_Personal_Info.setObjectName("pushButton_Personal_Info")
        self.pushButton_model = QtWidgets.QPushButton(self.page_Home)
        self.pushButton_model.setGeometry(QtCore.QRect(420, 350, 136, 36))
        self.pushButton_model.setStyleSheet("QPushButton {  \n"
"                background-color: #e6e6fa; /* 浅紫色背景 */  \n"
"                color: #ffffff; /* 白色字体 */  \n"
"                border: none; /* 移除边框 */  \n"
"                padding: 10px 20px; /* 内边距 */  \n"
"                text-align: center; /* 文本居中 */  \n"
"                font-size: 16px; /* 字体大小 */  \n"
"                cursor: pointer; /* 鼠标悬停时显示指针 */  \n"
"            }  \n"
"            QPushButton:hover {  \n"
"                background-color: #d1d1f0; /* 鼠标悬停时的背景色 */  \n"
"            }  \n"
"            QPushButton:pressed {  \n"
"                background-color: #b3b3e6; /* 按钮按下时的背景色 */  \n"
"            }  ")
        self.pushButton_model.setObjectName("pushButton_model")
        self.label = QtWidgets.QLabel(self.page_Home)
        self.label.setGeometry(QtCore.QRect(70, 10, 251, 101))
        self.label.setStyleSheet("\n"
"color: rgb(255, 170, 255);\n"
"font: 48pt \"华文行楷\";")
        self.label.setObjectName("label")
        self.label_8 = QtWidgets.QLabel(self.page_Home)
        self.label_8.setGeometry(QtCore.QRect(60, 170, 231, 211))
        self.label_8.setStyleSheet("image: url(:/icons/icon/喜鹊.png);")
        self.label_8.setText("")
        self.label_8.setObjectName("label_8")
        self.stackedWidget.addWidget(self.page_Home)
        self.page_Mmodel = QtWidgets.QWidget()
        self.page_Mmodel.setObjectName("page_Mmodel")
        self.tableWidget_4 = QtWidgets.QTableWidget(self.page_Mmodel)
        self.tableWidget_4.setGeometry(QtCore.QRect(10, 80, 571, 231))
        self.tableWidget_4.setObjectName("tableWidget_4")
        self.tableWidget_4.setColumnCount(9)
        self.tableWidget_4.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(8, item)
        self.pushButton_look_4 = QtWidgets.QPushButton(self.page_Mmodel)
        self.pushButton_look_4.setGeometry(QtCore.QRect(50, 370, 93, 28))
        self.pushButton_look_4.setObjectName("pushButton_look_4")
        self.pushButton_modify_2 = QtWidgets.QPushButton(self.page_Mmodel)
        self.pushButton_modify_2.setGeometry(QtCore.QRect(340, 370, 93, 28))
        self.pushButton_modify_2.setObjectName("pushButton_modify_2")
        self.label_9 = QtWidgets.QLabel(self.page_Mmodel)
        self.label_9.setGeometry(QtCore.QRect(20, 20, 171, 31))
        self.label_9.setStyleSheet("font: 15pt \"幼圆\";")
        self.label_9.setObjectName("label_9")
        self.stackedWidget.addWidget(self.page_Mmodel)
        self.page_Personal_Info = QtWidgets.QWidget()
        self.page_Personal_Info.setObjectName("page_Personal_Info")
        self.gridLayout = QtWidgets.QGridLayout(self.page_Personal_Info)
        self.gridLayout.setObjectName("gridLayout")
        self.lineEdit_Email = QtWidgets.QLineEdit(self.page_Personal_Info)
        self.lineEdit_Email.setText("")
        self.lineEdit_Email.setObjectName("lineEdit_Email")
        self.gridLayout.addWidget(self.lineEdit_Email, 1, 3, 1, 1)
        self.pushButton_look = QtWidgets.QPushButton(self.page_Personal_Info)
        self.pushButton_look.setMinimumSize(QtCore.QSize(0, 28))
        self.pushButton_look.setStyleSheet("background-color: rgb(15, 15, 15);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_look.setObjectName("pushButton_look")
        self.gridLayout.addWidget(self.pushButton_look, 3, 1, 1, 1)
        self.lineEdit_Username = QtWidgets.QLineEdit(self.page_Personal_Info)
        self.lineEdit_Username.setText("")
        self.lineEdit_Username.setObjectName("lineEdit_Username")
        self.gridLayout.addWidget(self.lineEdit_Username, 0, 3, 1, 1)
        self.lineEdit_PasswordHash = QtWidgets.QLineEdit(self.page_Personal_Info)
        self.lineEdit_PasswordHash.setText("")
        self.lineEdit_PasswordHash.setObjectName("lineEdit_PasswordHash")
        self.gridLayout.addWidget(self.lineEdit_PasswordHash, 1, 1, 1, 1)
        self.lineEdit_RegistrationDate = QtWidgets.QLineEdit(self.page_Personal_Info)
        self.lineEdit_RegistrationDate.setText("")
        self.lineEdit_RegistrationDate.setObjectName("lineEdit_RegistrationDate")
        self.gridLayout.addWidget(self.lineEdit_RegistrationDate, 2, 3, 1, 1)
        self.lineEdit_Gender = QtWidgets.QLineEdit(self.page_Personal_Info)
        self.lineEdit_Gender.setText("")
        self.lineEdit_Gender.setObjectName("lineEdit_Gender")
        self.gridLayout.addWidget(self.lineEdit_Gender, 2, 1, 1, 1)
        self.pushButton_modify = QtWidgets.QPushButton(self.page_Personal_Info)
        self.pushButton_modify.setStyleSheet("background-color: rgb(4, 4, 4);\n"
"color: rgb(255, 255, 255);")
        self.pushButton_modify.setObjectName("pushButton_modify")
        self.gridLayout.addWidget(self.pushButton_modify, 3, 3, 1, 1)
        self.lineEdit_UserID = QtWidgets.QLineEdit(self.page_Personal_Info)
        self.lineEdit_UserID.setText("")
        self.lineEdit_UserID.setObjectName("lineEdit_UserID")
        self.gridLayout.addWidget(self.lineEdit_UserID, 0, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.page_Personal_Info)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 2, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.page_Personal_Info)
        self.label_6.setStyleSheet("font: 10pt \"隶书\";")
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.page_Personal_Info)
        self.label_11.setObjectName("label_11")
        self.gridLayout.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.page_Personal_Info)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 2, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.page_Personal_Info)
        self.label_13.setObjectName("label_13")
        self.gridLayout.addWidget(self.label_13, 1, 2, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.page_Personal_Info)
        self.label_14.setObjectName("label_14")
        self.gridLayout.addWidget(self.label_14, 2, 2, 1, 1)
        self.stackedWidget.addWidget(self.page_Personal_Info)
        self.page_Experiment_Info = QtWidgets.QWidget()
        self.page_Experiment_Info.setObjectName("page_Experiment_Info")
        self.tableWidget = QtWidgets.QTableWidget(self.page_Experiment_Info)
        self.tableWidget.setGeometry(QtCore.QRect(10, 110, 561, 171))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.pushButton_look_1 = QtWidgets.QPushButton(self.page_Experiment_Info)
        self.pushButton_look_1.setGeometry(QtCore.QRect(80, 330, 93, 28))
        self.pushButton_look_1.setObjectName("pushButton_look_1")
        self.pushButton_delete = QtWidgets.QPushButton(self.page_Experiment_Info)
        self.pushButton_delete.setGeometry(QtCore.QRect(340, 330, 93, 28))
        self.pushButton_delete.setObjectName("pushButton_delete")
        self.label_10 = QtWidgets.QLabel(self.page_Experiment_Info)
        self.label_10.setGeometry(QtCore.QRect(30, 20, 161, 51))
        self.label_10.setStyleSheet("font: 15pt \"幼圆\";")
        self.label_10.setObjectName("label_10")
        self.stackedWidget.addWidget(self.page_Experiment_Info)
        self.page_Admin = QtWidgets.QWidget()
        self.page_Admin.setObjectName("page_Admin")
        self.label_2 = QtWidgets.QLabel(self.page_Admin)
        self.label_2.setGeometry(QtCore.QRect(10, 0, 211, 111))
        self.label_2.setStyleSheet("font: 24pt \"Agency FB\";")
        self.label_2.setObjectName("label_2")
        self.pushButton_Manage_Profile = QtWidgets.QPushButton(self.page_Admin)
        self.pushButton_Manage_Profile.setGeometry(QtCore.QRect(30, 100, 151, 71))
        self.pushButton_Manage_Profile.setObjectName("pushButton_Manage_Profile")
        self.pushButton_Manage_Experiments = QtWidgets.QPushButton(self.page_Admin)
        self.pushButton_Manage_Experiments.setGeometry(QtCore.QRect(310, 100, 141, 71))
        self.pushButton_Manage_Experiments.setObjectName("pushButton_Manage_Experiments")
        self.stackedWidget.addWidget(self.page_Admin)
        self.page_Manage_Profile = QtWidgets.QWidget()
        self.page_Manage_Profile.setObjectName("page_Manage_Profile")
        self.tableWidget_3 = QtWidgets.QTableWidget(self.page_Manage_Profile)
        self.tableWidget_3.setGeometry(QtCore.QRect(30, 80, 531, 261))
        self.tableWidget_3.setObjectName("tableWidget_3")
        self.tableWidget_3.setColumnCount(6)
        self.tableWidget_3.setRowCount(0)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(5, item)
        self.pushButton_look_3 = QtWidgets.QPushButton(self.page_Manage_Profile)
        self.pushButton_look_3.setGeometry(QtCore.QRect(70, 380, 93, 28))
        self.pushButton_look_3.setObjectName("pushButton_look_3")
        self.pushButton_delete_3 = QtWidgets.QPushButton(self.page_Manage_Profile)
        self.pushButton_delete_3.setGeometry(QtCore.QRect(380, 380, 93, 28))
        self.pushButton_delete_3.setObjectName("pushButton_delete_3")
        self.stackedWidget.addWidget(self.page_Manage_Profile)
        self.page_Manage_Experiments = QtWidgets.QWidget()
        self.page_Manage_Experiments.setObjectName("page_Manage_Experiments")
        self.tableWidget_2 = QtWidgets.QTableWidget(self.page_Manage_Experiments)
        self.tableWidget_2.setGeometry(QtCore.QRect(0, 130, 571, 192))
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(5)
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
        self.pushButton_look_2 = QtWidgets.QPushButton(self.page_Manage_Experiments)
        self.pushButton_look_2.setGeometry(QtCore.QRect(70, 380, 93, 28))
        self.pushButton_look_2.setObjectName("pushButton_look_2")
        self.pushButton_delete_2 = QtWidgets.QPushButton(self.page_Manage_Experiments)
        self.pushButton_delete_2.setGeometry(QtCore.QRect(320, 380, 93, 28))
        self.pushButton_delete_2.setObjectName("pushButton_delete_2")
        self.stackedWidget.addWidget(self.page_Manage_Experiments)
        self.page_Model = QtWidgets.QWidget()
        self.page_Model.setObjectName("page_Model")
        self.comboBox = QtWidgets.QComboBox(self.page_Model)
        self.comboBox.setGeometry(QtCore.QRect(290, 40, 171, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.stackedWidget_2 = QtWidgets.QStackedWidget(self.page_Model)
        self.stackedWidget_2.setGeometry(QtCore.QRect(340, 130, 193, 231))
        self.stackedWidget_2.setObjectName("stackedWidget_2")
        self.page_resnet = QtWidgets.QWidget()
        self.page_resnet.setObjectName("page_resnet")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.page_resnet)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.lineEdit_size = QtWidgets.QLineEdit(self.page_resnet)
        self.lineEdit_size.setText("")
        self.lineEdit_size.setObjectName("lineEdit_size")
        self.verticalLayout_5.addWidget(self.lineEdit_size)
        self.lineEdit_rate = QtWidgets.QLineEdit(self.page_resnet)
        self.lineEdit_rate.setText("")
        self.lineEdit_rate.setObjectName("lineEdit_rate")
        self.verticalLayout_5.addWidget(self.lineEdit_rate)
        self.comboBox_youhuaqi = QtWidgets.QComboBox(self.page_resnet)
        self.comboBox_youhuaqi.setObjectName("comboBox_youhuaqi")
        self.comboBox_youhuaqi.addItem("")
        self.comboBox_youhuaqi.addItem("")
        self.verticalLayout_5.addWidget(self.comboBox_youhuaqi)
        self.stackedWidget_2.addWidget(self.page_resnet)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.page_3)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.lineEdit_max_depth = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_max_depth.setObjectName("lineEdit_max_depth")
        self.verticalLayout_6.addWidget(self.lineEdit_max_depth)
        self.lineEdit_max_leaf_nodes = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_max_leaf_nodes.setObjectName("lineEdit_max_leaf_nodes")
        self.verticalLayout_6.addWidget(self.lineEdit_max_leaf_nodes)
        self.lineEdit_random_2 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_random_2.setObjectName("lineEdit_random_2")
        self.verticalLayout_6.addWidget(self.lineEdit_random_2)
        self.stackedWidget_2.addWidget(self.page_3)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.page_2)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.lineEdit_C = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_C.setObjectName("lineEdit_C")
        self.verticalLayout_4.addWidget(self.lineEdit_C)
        self.lineEdit = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_4.addWidget(self.lineEdit)
        self.lineEdit_random = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit_random.setObjectName("lineEdit_random")
        self.verticalLayout_4.addWidget(self.lineEdit_random)
        self.stackedWidget_2.addWidget(self.page_2)
        self.pushButton_data_directory = QtWidgets.QPushButton(self.page_Model)
        self.pushButton_data_directory.setGeometry(QtCore.QRect(110, 160, 93, 28))
        self.pushButton_data_directory.setObjectName("pushButton_data_directory")
        self.pushButton_model_directory = QtWidgets.QPushButton(self.page_Model)
        self.pushButton_model_directory.setGeometry(QtCore.QRect(110, 230, 93, 28))
        self.pushButton_model_directory.setObjectName("pushButton_model_directory")
        self.pushButton_training = QtWidgets.QPushButton(self.page_Model)
        self.pushButton_training.setGeometry(QtCore.QRect(110, 310, 93, 28))
        self.pushButton_training.setObjectName("pushButton_training")
        self.lineEdit_accuracy = QtWidgets.QLineEdit(self.page_Model)
        self.lineEdit_accuracy.setGeometry(QtCore.QRect(400, 380, 101, 31))
        self.lineEdit_accuracy.setObjectName("lineEdit_accuracy")
        self.label_7 = QtWidgets.QLabel(self.page_Model)
        self.label_7.setGeometry(QtCore.QRect(340, 380, 51, 31))
        self.label_7.setObjectName("label_7")
        self.label_16 = QtWidgets.QLabel(self.page_Model)
        self.label_16.setGeometry(QtCore.QRect(20, 10, 201, 151))
        self.label_16.setStyleSheet("image: url(:/icons/icon/锦鲤戏水.png);")
        self.label_16.setText("")
        self.label_16.setObjectName("label_16")
        self.stackedWidget.addWidget(self.page_Model)
        self.page_Experiments = QtWidgets.QWidget()
        self.page_Experiments.setObjectName("page_Experiments")
        self.pushButton_5 = QtWidgets.QPushButton(self.page_Experiments)
        self.pushButton_5.setGeometry(QtCore.QRect(140, 150, 93, 28))
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self.page_Experiments)
        self.pushButton_6.setGeometry(QtCore.QRect(140, 220, 93, 28))
        self.pushButton_6.setObjectName("pushButton_6")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.page_Experiments)
        self.lineEdit_2.setGeometry(QtCore.QRect(360, 230, 71, 31))
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.label_4 = QtWidgets.QLabel(self.page_Experiments)
        self.label_4.setGeometry(QtCore.QRect(290, 230, 72, 31))
        self.label_4.setObjectName("label_4")
        self.pushButton_7 = QtWidgets.QPushButton(self.page_Experiments)
        self.pushButton_7.setGeometry(QtCore.QRect(140, 330, 93, 28))
        self.pushButton_7.setObjectName("pushButton_7")
        self.label_3 = QtWidgets.QLabel(self.page_Experiments)
        self.label_3.setGeometry(QtCore.QRect(330, 90, 101, 81))
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_15 = QtWidgets.QLabel(self.page_Experiments)
        self.label_15.setGeometry(QtCore.QRect(0, 10, 171, 121))
        self.label_15.setStyleSheet("image: url(:/icons/辞岁女孩.png);")
        self.label_15.setText("")
        self.label_15.setObjectName("label_15")
        self.label_17 = QtWidgets.QLabel(self.page_Experiments)
        self.label_17.setGeometry(QtCore.QRect(10, 0, 141, 111))
        self.label_17.setStyleSheet("image: url(:/icons/icon/锦鲤游玩.png);")
        self.label_17.setText("")
        self.label_17.setObjectName("label_17")
        self.label_18 = QtWidgets.QLabel(self.page_Experiments)
        self.label_18.setGeometry(QtCore.QRect(390, 320, 171, 131))
        self.label_18.setStyleSheet("image: url(:/icons/icon/辞岁女孩.png);")
        self.label_18.setText("")
        self.label_18.setObjectName("label_18")
        self.label_19 = QtWidgets.QLabel(self.page_Experiments)
        self.label_19.setGeometry(QtCore.QRect(0, 330, 141, 131))
        self.label_19.setStyleSheet("image: url(:/icons/icon/辞岁男孩.png);")
        self.label_19.setText("")
        self.label_19.setObjectName("label_19")
        self.stackedWidget.addWidget(self.page_Experiments)
        self.verticalLayout_3.addWidget(self.stackedWidget)
        self.horizontalLayout_4.addWidget(self.frame_7)
        self.verticalLayout.addWidget(self.frame_3)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.stackedWidget_2.setCurrentIndex(0)
        self.pushButton_Close2.clicked.connect(MainWindow.close) # type: ignore
        self.pushButton_Minimize.clicked.connect(MainWindow.showMinimized) # type: ignore
        self.comboBox.currentIndexChanged['int'].connect(self.stackedWidget_2.setCurrentIndex) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "机器学习实验管理平台"))
        self.pushButton_Minimize.setText(_translate("MainWindow", "缩小"))
        self.pushButton_Close2.setText(_translate("MainWindow", "关闭"))
        self.pushButton_Home.setText(_translate("MainWindow", "首页"))
        self.pushButton_Model.setText(_translate("MainWindow", "模型训练"))
        self.pushButton_Experiments.setText(_translate("MainWindow", "实验"))
        self.pushButton_Experiment_Info.setText(_translate("MainWindow", "查询实验信息"))
        self.pushButton_Personal_Info.setText(_translate("MainWindow", "查询个人信息"))
        self.pushButton_model.setText(_translate("MainWindow", "查询模型信息"))
        self.label.setText(_translate("MainWindow", "welcome"))
        item = self.tableWidget_4.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "模型名"))
        item = self.tableWidget_4.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "评估效果"))
        item = self.tableWidget_4.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "数据集路径"))
        item = self.tableWidget_4.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "模型路径"))
        item = self.tableWidget_4.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "模型参数"))
        self.pushButton_look_4.setText(_translate("MainWindow", "查看"))
        self.pushButton_modify_2.setText(_translate("MainWindow", "修改"))
        self.label_9.setText(_translate("MainWindow", "查看模型信息"))
        self.pushButton_look.setText(_translate("MainWindow", "查看"))
        self.pushButton_modify.setText(_translate("MainWindow", "修改"))
        self.label_5.setText(_translate("MainWindow", "   用户名："))
        self.label_6.setText(_translate("MainWindow", "ID:"))
        self.label_11.setText(_translate("MainWindow", "密码："))
        self.label_12.setText(_translate("MainWindow", "性别："))
        self.label_13.setText(_translate("MainWindow", "    邮箱："))
        self.label_14.setText(_translate("MainWindow", "   注册时间："))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "实验ID:"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "实验名："))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "实验状态"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "实验时间"))
        self.pushButton_look_1.setText(_translate("MainWindow", "查询"))
        self.pushButton_delete.setText(_translate("MainWindow", "删除"))
        self.label_10.setText(_translate("MainWindow", "查看实验信息"))
        self.label_2.setText(_translate("MainWindow", "管理员页面"))
        self.pushButton_Manage_Profile.setText(_translate("MainWindow", "管理个人账户"))
        self.pushButton_Manage_Experiments.setText(_translate("MainWindow", "管理实验信息"))
        item = self.tableWidget_3.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "用户ID"))
        item = self.tableWidget_3.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "用户名"))
        item = self.tableWidget_3.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "密码"))
        item = self.tableWidget_3.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "注册时间"))
        item = self.tableWidget_3.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "邮箱"))
        item = self.tableWidget_3.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "性别"))
        self.pushButton_look_3.setText(_translate("MainWindow", "查询"))
        self.pushButton_delete_3.setText(_translate("MainWindow", "删除"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "用户ID"))
        item = self.tableWidget_2.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "实验ID"))
        item = self.tableWidget_2.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "实验状态"))
        item = self.tableWidget_2.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "实验状态"))
        item = self.tableWidget_2.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "实验时间"))
        self.pushButton_look_2.setText(_translate("MainWindow", "查询"))
        self.pushButton_delete_2.setText(_translate("MainWindow", "修改"))
        self.comboBox.setItemText(0, _translate("MainWindow", "CNN"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Random_Forest"))
        self.comboBox.setItemText(2, _translate("MainWindow", "RESNET"))
        self.lineEdit_size.setPlaceholderText(_translate("MainWindow", "批量："))
        self.lineEdit_rate.setPlaceholderText(_translate("MainWindow", "学习率："))
        self.comboBox_youhuaqi.setItemText(0, _translate("MainWindow", "SGD"))
        self.comboBox_youhuaqi.setItemText(1, _translate("MainWindow", "Adam"))
        self.lineEdit_max_depth.setPlaceholderText(_translate("MainWindow", "n_estimators"))
        self.lineEdit_max_leaf_nodes.setPlaceholderText(_translate("MainWindow", "max_features"))
        self.lineEdit_random_2.setPlaceholderText(_translate("MainWindow", "max_depth"))
        self.lineEdit_C.setPlaceholderText(_translate("MainWindow", "IR："))
        self.lineEdit.setPlaceholderText(_translate("MainWindow", "bath_size:"))
        self.lineEdit_random.setPlaceholderText(_translate("MainWindow", "epochs:"))
        self.pushButton_data_directory.setText(_translate("MainWindow", "上传数据集"))
        self.pushButton_model_directory.setText(_translate("MainWindow", "模型保存"))
        self.pushButton_training.setText(_translate("MainWindow", "运行"))
        self.label_7.setText(_translate("MainWindow", "准确率："))
        self.pushButton_5.setText(_translate("MainWindow", "上传图片"))
        self.pushButton_6.setText(_translate("MainWindow", "选择模型"))
        self.label_4.setText(_translate("MainWindow", "结果显示："))
        self.pushButton_7.setText(_translate("MainWindow", "预测"))
import res_rc
