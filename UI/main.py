from LoginUi import Ui_LoginWindow  # 确保正确导入 Ui_LoginWindow
from Interface import Ui_MainWindow  # 确保正确导入 Ui_MainWindow
from Manager import Ui_ManageWindow
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QMessageBox,QFileDialog, QProgressDialog
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
import mysql.connector
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import warnings
import torchvision.models as models
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import json



warnings.filterwarnings("ignore", category=DeprecationWarning)


class ManageWindow(QMainWindow):
    def __init__(self):
        super(ManageWindow, self).__init__()  # 明确指定类名
        self.ui = Ui_ManageWindow()  # 使用 Ui_LoginWindow
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.ui.pushButton_look_2.clicked.connect(self.load_all_experiments)
        self.ui.pushButton_look.clicked.connect(self.load_all_personal_info)
        self.ui.pushButton_delect_2.clicked.connect(self.delete_all_experiment)
        self.ui.pushButton_delect.clicked.connect(self.delete_all_personal_info)
        self.ui.pushButton_Admin.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_Experiments.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_modify.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_modify_2.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.pushButton_look_3.clicked.connect( self.view_all_personal)
        self.ui.pushButton_look_4.clicked.connect(self.view_all_experiment)
        self.ui.pushButton_modify_3.clicked.connect(self.modify_all_personal)
        self.ui.pushButton_modify_4.clicked.connect(self.modify_all_experiments)

    #修改实验信息
    def modify_all_experiments(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                ResultId = int(self.ui.lineEdit_experiment_ID.text())
                userid = int(self.ui.lineEdit_user_ID.text())  # Assuming UserID is an integer
                modelname = self.ui.lineEdit_model.text()
                predictiondate = self.ui.lineEdit_experiment_date.text()
                predictiondatapath = self.ui.lineEdit_experiment_image.text()
                predictionresult = self.ui.lineEdit_experiment_result.text()

                # 使用参数化查询来执行更新操作，确保数据安全
                cursor.execute("""
                                UPDATE experimentresults
                                SET UserID= %s, ModelName= %s, PredictionDate= %s, PredictionDataPath= %s, PredictionResult= %s
                                WHERE ResultId = %s
                                """,
                               (userid, modelname, predictiondate, predictiondatapath, predictionresult, ResultId))

                conn.commit()
                print("实验信息更新成功。")

                # 更新界面显示的用户信息
                self.ui.lineEdit_user_ID.setText(str(userid))
                self.ui.lineEdit_model.setText(modelname)
                self.ui.lineEdit_experiment_date.setText(predictiondate)
                self.ui.lineEdit_experiment_image.setText(predictiondatapath)
                self.ui.lineEdit_experiment_result.setText(predictionresult)

            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)

    #修改个人信息
    def modify_all_personal(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                user_id = self.ui.lineEdit_ID.text()
                email = self.ui.lineEdit_Email.text()
                gender = self.ui.lineEdit_gender.text()
                password_hash = self.ui.lineEdit_password.text()
                registration_date = self.ui.lineEdit_date.text()
                username = self.ui.lineEdit_name.text()

                cursor.execute("""
                                    UPDATE Users
                                    SET Email = %s, Gender = %s, PasswordHash = %s, RegistrationDate = %s, Username = %s
                                    WHERE UserID = %s
                                """, (email, gender, password_hash, registration_date, username, user_id))

                conn.commit()
                print("用户信息更新成功。")

                # 更新界面显示的用户信息
                self.ui.lineEdit_Email.setText(email)
                self.ui.lineEdit_gender.setText(gender)
                self.ui.lineEdit_password.setText(password_hash)
                self.ui.lineEdit_date.setText(registration_date)
                self.ui.lineEdit_name.setText(username)

            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)

     #查看所有人的实验信息（表格）
    def load_all_experiments(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT UserID,ResultID  ,ModelName, PredictionDate ,PredictionDataPath ,PredictionResult "
                    "FROM Experimentresults",
                )
                rows = cursor.fetchall()

                if rows:
                    self.ui.tableWidget_2.setRowCount(len(rows))
                    self.ui.tableWidget_2.setColumnCount(6)
                    self.ui.tableWidget_2.setHorizontalHeaderLabels(
                        ['用户ID', '实验ID', '模型名', '实验时间', '预测数据路径', '实验结果'])

                    for row_num, row_data in enumerate(rows):
                        for col_num, data in enumerate(row_data):
                            self.ui.tableWidget_2.setItem(row_num, col_num, QTableWidgetItem(str(data)))
                else:
                    print("没有找到实验信息。")
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)

    #查看个人信息
    def view_all_personal(self):
        conn = self.connect_to_database()
        UserID=int(self.ui.lineEdit_ID.text())
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT Email, Gender, PasswordHash, RegistrationDate, UserID, Username FROM Users WHERE UserID = %s",
                    (UserID,))
                row = cursor.fetchone()

                if row:
                    email, gender, password_hash, registration_date, user_id, username = row
                    self.ui.lineEdit_Email.setText(email)
                    self.ui.lineEdit_gender.setText(gender)
                    self.ui.lineEdit_password.setText(password_hash)
                    self.ui.lineEdit_date.setText(str(registration_date))
                    self.ui.lineEdit_name.setText(username)
                else:
                    print("用户未找到。")
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)

    #查看个人实验信息
    def view_all_experiment(self):
        conn = self.connect_to_database()
        ResultId=int(self.ui.lineEdit_experiment_ID.text())
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT UserID,  ModelName, PredictionDate ,PredictionDataPath ,PredictionResult "
                    "FROM Experimentresults "
                    "WHERE ResultId = %s ",
                    (ResultId,))
                rows = cursor.fetchone()

                if rows:
                    userid, modelname, predictiondate, predictiondatapath, predictionresult = rows
                    # self.ui.lineEdit_experiment_ID.setText(reslutid)
                    self.ui.lineEdit_user_ID.setText(str(userid))
                    self.ui.lineEdit_model.setText(modelname)
                    self.ui.lineEdit_experiment_date.setText(str(predictiondate))
                    self.ui.lineEdit_experiment_image.setText(predictiondatapath)
                    self.ui.lineEdit_experiment_result.setText(predictionresult)
                else:
                    print("没有找到实验信息。")
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)

    #一键删除个人实验信息
    def delete_all_experiment(self):
        selected_row = self.ui.tableWidget_2.currentRow()
        if selected_row != -1:
            experiment_id = self.ui.tableWidget_2.item(selected_row, 1).text()  # 获取选中行的实验ID
            conn = self.connect_to_database()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM Experimentresults WHERE resultID = %s", (experiment_id,))
                    conn.commit()
                    print(f"实验ID为 {experiment_id} 的实验信息删除成功。")
                    self.ui.tableWidget_2.removeRow(selected_row)  # 删除表格中的行
                except mysql.connector.Error as err:
                    print(f"数据库错误: {err}")
                finally:
                    self.close_database_connection(conn, cursor)
        else:
            QMessageBox.warning(self, "警告", "请先选择要删除的实验信息。", QMessageBox.Ok)

    #一键查询所有人的实验信息（文本框）
    def  load_all_personal_info(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT UserID, Username,PasswordHash,  RegistrationDate ,Email, Gender FROM users ",
                )
                rows = cursor.fetchall()

                if rows:
                    self.ui.tableWidget.setRowCount(len(rows))
                    self.ui.tableWidget.setColumnCount(6)
                    self.ui.tableWidget.setHorizontalHeaderLabels(
                        ['用户ID', '用户名', '密码', '注册时间', '邮箱', '性别'])

                    for row_num, row_data in enumerate(rows):
                        for col_num, data in enumerate(row_data):
                            self.ui.tableWidget.setItem(row_num, col_num, QTableWidgetItem(str(data)))
                else:
                    print("没有找到实验信息。")
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)

    #一键删除个人信息
    def delete_all_personal_info(self):
        selected_row = self.ui.tableWidget.currentRow()
        if selected_row != -1:
            user_id = self.ui.tableWidget.item(selected_row,
                                               0).text()
            conn = self.connect_to_database()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM users WHERE UserID = %s", (user_id,))
                    conn.commit()
                    print(f"用户ID为 {user_id} 的用户信息删除成功。")

                    self.ui.tableWidget.removeRow(selected_row)  # 删除表格中的行

                except mysql.connector.Error as err:
                    print(f"数据库错误: {err}")
                finally:
                    self.close_database_connection(conn, cursor)
        else:
            QMessageBox.warning(self, "警告", "请先选择要删除的用户信息。", QMessageBox.Ok)

    def connect_to_database(self):
        try:
            conn = mysql.connector.connect(
                host='127.0.0.1',
                database='machinelearn',
                user='root',
                password='123456'
            )
            return conn
        except mysql.connector.Error as err:
            print(f"数据库连接错误: {err}")
            return None

    def close_database_connection(self, conn, cursor):
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("数据库连接已关闭。")




class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(64 * 5 * 5, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class LoginWindow(QMainWindow):
    def __init__(self):
        super(LoginWindow, self).__init__()  # 明确指定类名
        self.ui = Ui_LoginWindow()  # 使用 Ui_LoginWindow
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)

        # 切换登录界面和注册界面
        self.ui.pushButton_Login.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(1))
        self.ui.pushButton_Register.clicked.connect(lambda: self.ui.stackedWidget_2.setCurrentIndex(0))

        # 关闭窗口
        self.ui.pushButton.clicked.connect(self.close)

        # 连接登录确认按钮的点击信号到登录函数
        self.ui.pushButton_L_sure.clicked.connect(self.login_in)
        # 连接注册按钮的点击信号到注册函数
        self.ui.pushButton_R_sure.clicked.connect(self.register_in)

        self.show()

    def connect_to_database(self):
        try:
            conn = mysql.connector.connect(
                host='127.0.0.1',
                database='machinelearn',
                user='root',
                password='123456'
            )
            return conn
        except mysql.connector.Error as err:
            print(f"数据库连接错误: {err}")
            return None

    def close_database_connection(self, conn, cursor):
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("数据库连接已关闭。")

    def login_in(self):
        accountid = self.ui.lineEdit_L_account.text()
        password = self.ui.lineEdit_L_password.text()

        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("SELECT UserID, PasswordHash, isadmin "
                               "FROM Users "
                               "WHERE UserID = %s", (accountid,))
                row = cursor.fetchone()

                if row is not None:
                    userID, password_hash, is_admin = row  # Unpack the values from the row
                    if password == password_hash:
                        if is_admin == 1:
                            self.main_window = ManageWindow()
                            self.main_window.show()
                            self.close()
                        else:
                            print("登录成功")
                            self.main_window = MainWindow(user_id=userID)  # Pass user_id to MainWindow
                            self.main_window.show()
                            self.close()  # Hide login window
                    else:
                        self.ui.stackedWidget.setCurrentIndex(2)  # Password incorrect
                else:
                    self.ui.stackedWidget.setCurrentIndex(1)  # User not found

            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")

            finally:
                self.close_database_connection(conn, cursor)

    def register_in(self):
        account = self.ui.lineEdit_R_account.text()
        password_1 = self.ui.lineEdit_R_password_1.text()
        password_2 = self.ui.lineEdit_R_password_2.text()
        email = self.ui.lineEdit_Email.text()

        if password_1 != password_2:
            self.ui.stackedWidget.setCurrentIndex(3)
            return

        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO Users (UserID, Email, PasswordHash) VALUES (%s, %s, %s)",
                               (account, email, password_1))
                conn.commit()
                self.ui.stackedWidget.setCurrentIndex(4)
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)


class MainWindow(QMainWindow):
    def __init__(self, user_id):
        super(MainWindow, self).__init__()
        self.user_id = user_id  # 存储 user_id
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowFlags(QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.data_directory = ""
        self.model_directory = ""
        self.prediction_image_path = ""
        self.trained_model_path = ""
        self.model = None  # 用于存储加载的模型
        self.model_type = ""  # 用于存储模型类型

        # 连接按钮点击事件到相应的函数
        self.ui.pushButton_Home.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(0))
        self.ui.pushButton_Model.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(7))
        self.ui.pushButton_Experiments.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(8))
        self.ui.pushButton_Personal_Info.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(2))
        self.ui.pushButton_Experiment_Info.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(3))
        self.ui.pushButton_Manage_Experiments.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(6))
        self.ui.pushButton_Manage_Profile.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(5))
        self.ui.pushButton_model.clicked.connect(lambda: self.ui.stackedWidget.setCurrentIndex(1))
        self.ui.pushButton_look.clicked.connect(self.load_personal_info)
        self.ui.pushButton_modify.clicked.connect(self.modify_personal_info)
        self.ui.pushButton_look_1.clicked.connect(self.load_experiments)
        self.ui.pushButton_delete.clicked.connect(self.delete_experiment)
        self.ui.pushButton_data_directory.clicked.connect(self.select_data_directory)
        self.ui.pushButton_model_directory.clicked.connect(self.select_model_directory)
        self.ui.pushButton_training.clicked.connect(self.start_training)
        self.ui.pushButton_5.clicked.connect(self.select_image_for_prediction)
        self.ui.pushButton_6.clicked.connect(self.select_trained_model)
        self.ui.pushButton_7.clicked.connect(self.predict_using_model)
        self.ui.pushButton_look_4.clicked.connect(self.look_model)
        self.ui.pushButton_modify_2.clicked.connect(self.modify_model)



        self.show()


    def select_data_directory(self):
        options = QFileDialog.Options()
        self.data_directory = QFileDialog.getExistingDirectory(self, "选择数据集文件夹", options=options)
        if self.data_directory:
            print(f"选择的数据集文件夹路径: {self.data_directory}")

    def select_model_directory(self):
        options = QFileDialog.Options()
        self.model_directory = QFileDialog.getExistingDirectory(self, "选择模型保存文件夹", options=options)
        if self.model_directory:
            print(f"选择的模型保存文件夹路径: {self.model_directory}")


    def start_training(self):
        if not self.data_directory or not self.model_directory:
            print("请先选择数据集文件夹和模型保存文件夹")
            return
        size_1 = int(self.ui.lineEdit.text()) if self.ui.lineEdit.text() else 64
        IR_1 = float(self.ui.lineEdit_C.text()) if self.ui.lineEdit_C.text() else 0.001
        epochs = int(self.ui.lineEdit_random.text())if self.ui.lineEdit_random.text() else 10

        size = int(self.ui.lineEdit_size.text()) if self.ui.lineEdit_size.text() else 64
        IR = float(self.ui.lineEdit_rate.text()) if self.ui.lineEdit_rate.text() else 0.001
        youhuaqi_index = self.ui.comboBox_youhuaqi.currentIndex()
        youhuaqi = 'SGD' if youhuaqi_index == 0 else 'ADAM'

        max_depth = int(self.ui.lineEdit_max_depth.text()) if self.ui.lineEdit_max_depth.text() else None
        n_estimators = int(self.ui.lineEdit_max_leaf_nodes.text()) if self.ui.lineEdit_max_leaf_nodes.text() else 100
        random_state = int(self.ui.lineEdit_random_2.text()) if self.ui.lineEdit_random_2.text() else None

        model_type_index = self.ui.comboBox.currentIndex()

        self.progress = QProgressDialog("模型正在训练...", "取消", 0, 100, self)
        self.progress.setWindowTitle("模型训练进度")
        self.progress.setWindowModality(QtCore.Qt.WindowModal)
        self.progress.setAutoClose(True)
        self.progress.setAutoReset(True)

        if model_type_index == 0:
            self.train_cnn(size, IR, youhuaqi)
        elif model_type_index == 1:
            self.train_Random_Forest(n_estimators, max_depth, random_state)
        elif model_type_index == 2:
            self.train_ResNet(size_1, IR_1, epochs)
        else:
            print("未识别的模型类型")

    def train_ResNet(self, size_1, IR_1, epochs):
        print("Training ResNet model...")

        # Data preprocessing
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Load CIFAR-10 dataset
        trainset = CIFAR10(root=self.data_directory, train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=size_1, shuffle=True, num_workers=2)
        testset = CIFAR10(root=self.data_directory, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=size_1, shuffle=False, num_workers=2)

        # Initialize model and optimizer
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(pretrained=False, num_classes=10).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=IR_1, momentum=0.9)

        # Training the model
        num_epochs = epochs
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.progress.setValue(int(100 * (epoch * len(trainloader) + i + 1) / (num_epochs * len(trainloader))))
                if self.progress.wasCanceled():
                    return
            print(f'Epoch {epoch + 1} loss: {running_loss / (i + 1)}')


        print('Finished Training')

        # Testing the model
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy of the model on the test images: {accuracy}%')

        # Save the model
        model_path = os.path.join(self.model_directory, 'cifar10_resnet.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
        self.progress.setValue(100)
        self.ui.lineEdit_accuracy.setText(f"{accuracy:.2f}%")
        conn = self.connect_to_database()
        if conn:
            try:
                cur = conn.cursor()
                insert_stmt = (
                    "INSERT INTO Train (ModelName, ModelEval, DataPath, ModelPath, ModelParameters, UserId) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                )

                cur.execute(insert_stmt, (
                    'RESNET',  # 模型名称
                    f"{accuracy :.2f}%",  # 模型评估结果
                    self.data_directory,  # 数据集路径
                    model_path,  # 模型保存路径
                    json.dumps({'size_1': size_1, ' IR_1':  IR_1, 'epochs': epochs}),  # 模型参数信息转换为JSON格式
                    self.user_id,  # 用户ID
                ))
                conn.commit()
                QMessageBox.information(self, "模型训练完成", "RESNET模型训练完成，信息已保存到数据库。")
            except mysql.connector.Error as err:
                QMessageBox.critical(self, "数据库错误", f"发生错误：{err}")
            finally:
                self.close_database_connection(conn, cur)

    def train_cnn(self, size, IR, youhuaqi):
        print("Training CNN model...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        trainset = CIFAR10(root=self.data_directory, train=True, download=True, transform=transform)
        trainloader = DataLoader(trainset, batch_size=size, shuffle=True, num_workers=2)
        testset = CIFAR10(root=self.data_directory, train=False, download=True, transform=transform)
        testloader = DataLoader(testset, batch_size=size, shuffle=False, num_workers=2)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = CNN().to(device)
        criterion = nn.CrossEntropyLoss()

        if youhuaqi == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=IR, momentum=0.9)
        elif youhuaqi == 'ADAM':
            optimizer = optim.Adam(model.parameters(), lr=IR)
        else:
            print("未识别的优化器")
            return

        num_epochs = 20
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.progress.setValue(int(100 * (epoch * len(trainloader) + i + 1) / (num_epochs * len(trainloader))))
                if self.progress.wasCanceled():
                    return
            print(f'Epoch {epoch + 1} loss: {running_loss / (i + 1)}')



        print('Finished Training')

        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # 构建训练信息字典
        training_info = {}
        training_info['modelname'] = 'CNN'
        training_info['modeleval'] = f"{accuracy:.2f}%"
        training_info['Modelparaments'] = {
            'batchsize': size,
            'lr': IR,
            'optimizer': youhuaqi,
        }

        print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

        # UI 更新代码
        self.ui.lineEdit_accuracy.setText(f"{accuracy:.2f}%")

        model_path = os.path.join(self.model_directory, 'cifar10_cnn.pth')
        torch.save(model.state_dict(), model_path)
        print(f'Model saved to {model_path}')
        self.progress.setValue(100)

        # 连接数据库
        conn = self.connect_to_database()
        if conn:
            try:
                cur = conn.cursor()
                # 构建插入语句
                insert_stmt = (
                    "INSERT INTO Train (ModelName, ModelEval, DataPath, ModelPath, ModelParameters, UserId) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                )

                # 假设当前登录用户的ID存储在self.user_id
                cur.execute(insert_stmt, (
                    training_info['modelname'],
                    training_info['modeleval'],
                    self.data_directory,
                    model_path,
                    json.dumps(training_info['Modelparaments']),
                    self.user_id,
                ))
                conn.commit()
                QMessageBox.information(self, "模型训练完成", "模型训练完成，信息已保存到数据库。")
            except mysql.connector.Error as err:
                QMessageBox.critical(self, "数据库错误", f"发生错误：{err}")
            finally:
                # 关闭数据库连接
                self.close_database_connection(conn, cur)

    def train_Random_Forest(self, n_estimators, max_depth, random_state):
        print("Training Decision Tree model...")

        # 数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 加载CIFAR-10数据集
        trainset = CIFAR10(root=self.data_directory, train=True, download=True, transform=transform)
        testset = CIFAR10(root=self.data_directory, train=False, download=True, transform=transform)

        # 将数据转换为适用于决策树的格式
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        for data in trainset:
            image, label = data
            X_train.append(image.view(-1).numpy())
            y_train.append(label)

        for data in testset:
            image, label = data
            X_test.append(image.view(-1).numpy())
            y_test.append(label)

        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        clf.fit(X_train, y_train)

        # 在测试集上评估模型性能
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f'Accuracy of the Decision Tree model on the test images: {accuracy * 100:.2f}%')
        self.ui.lineEdit_accuracy.setText(f"{accuracy * 100:.2f}%")

        # 保存模型部分基本保持不变，但建议更改模型文件名以反映其是随机森林模型
        model_path = os.path.join(self.model_directory, 'cifar10_random_forest_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print(f'Random Forest model saved to {model_path}')

        # 插入到数据库
        conn = self.connect_to_database()
        if conn:
            try:
                cur = conn.cursor()
                # 构建插入语句
                insert_stmt = (
                    "INSERT INTO Train (ModelName, ModelEval, DataPath, ModelPath, ModelParameters, UserId) "
                    "VALUES (%s, %s, %s, %s, %s, %s)"
                )

                # 假设当前登录用户的ID存储在self.user_id
                cur.execute(insert_stmt, (
                    'RANDOM_FOREST',  # 固定模型名称
                    f"{accuracy * 100:.2f}%",  # 模型评估结果
                    self.data_directory,  # 数据集路径
                    model_path,  # 模型保存路径
                    json.dumps({'n_estimators': n_estimators, 'max_depth': max_depth, 'random_state': random_state}),
                    # 将模型参数信息转换为JSON格式
                    self.user_id,
                ))
                conn.commit()
                QMessageBox.information(self, "模型训练完成", "随机森林模型训练完成，信息已保存到数据库。")
            except mysql.connector.Error as err:
                QMessageBox.critical(self, "数据库错误", f"发生错误：{err}")
            finally:
                # 关闭数据库连接
                self.close_database_connection(conn, cur)

    def predict_using_model(self):
        if not self.model:
            print("请先加载模型")
            return

        if not self.prediction_image_path:
            print("请先选择要预测的图像")
            return

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # 数据预处理
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 确保图像大小为32x32，与CIFAR-10图像大小一致
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # 检查图像文件是否存在
        if not os.path.exists(self.prediction_image_path):
            print(f"Image path {self.prediction_image_path} does not exist.")
            return

        # 加载并预处理图像
        image = Image.open(self.prediction_image_path)
        image = transform(image).unsqueeze(0).to(device)  # 添加batch维度并转移到设备

        try:
            if self.model_type in ["cnn", "ResNet"]:
                # 将模型移动到相同的设备上
                self.model.to(device)

                # 使用 CNN 或 ResNet 模型进行预测
                with torch.no_grad():  # 不需要计算梯度，节省内存和计算资源
                    output = self.model(image)
                    _, predicted = torch.max(output, 1)
                predicted_class = classes[predicted.item()]  # 假设classes是你的类标签列表
            elif self.model_type == "random_forest":
                # 使用随机森林模型进行预测
                image_cpu = image.cpu()  # 将 CUDA 张量移动到 CPU 上
                image_numpy = image_cpu.view(-1).numpy()  # 将图像数据展平并转换为 NumPy 数组
                predicted_class = classes[self.model.predict([image_numpy])[0]]
            else:
                print(f"Unsupported model type: {self.model_type}")
                return
        except Exception as e:
            print(f"Error during prediction: {e}")
            return

        # 准备要插入数据库的数据
        model_name = self.model_type.upper()  # 模型名称
        prediction_data_path = self.prediction_image_path  # 预测图片路径
        prediction_result = predicted_class  # 预测结果
        user_id = self.user_id  # 用户ID

        # 连接数据库
        conn = self.connect_to_database()
        if conn:
            try:
                cur = conn.cursor()

                # 查询 train 表中是否存在相应的模型名称记录
                cur.execute("SELECT ModelName FROM Train WHERE ModelName = %s", (model_name,))
                result = cur.fetchone()
                if result:
                    model_name_found = result[0]
                    print("Found model in train table:", model_name_found)
                    # 这里可以根据需要继续处理查询结果

                if result:
                    # 构建插入语句
                    insert_stmt = (
                        "INSERT INTO experimentresults (modelname, predictiondatapath, predictionresult, userid) "
                        "VALUES (%s, %s, %s, %s)"
                    )

                    cur.execute(insert_stmt, (
                        model_name,
                        prediction_data_path,
                        prediction_result,
                        user_id,
                    ))
                    conn.commit()
                    print("Prediction result saved to database.")
                else:
                    print(f"Model name {model_name} not found in Train table.")
            except mysql.connector.Error as err:
                print(f"Database error: {err}")
            finally:
                self.close_database_connection(conn, cur)

            # 输出预测结果
            print(f'Predicted class: {predicted_class}')

            # 将预测结果显示在 lineEdit_2 中
            self.ui.lineEdit_2.setText(predicted_class)
    def select_image_for_prediction(self):
        options = QFileDialog.Options()
        self.prediction_image_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择预测图像",
            "",
            "Image Files (*.png *.jpg *.bmp)",
            options=options
        )
        if self.prediction_image_path:
            print(f"选择的预测图像路径: {self.prediction_image_path}")
            # 使用文件路径获取图片文件，并设置图片大小为 label 控件的大小
            img = QPixmap(self.prediction_image_path).scaled(
                self.ui.label_3.width(), self.ui.label_3.height()
            )

            # 在 label 控件上显示选择的图片
            self.ui.label_3.setPixmap(img)

    def select_trained_model(self):
        # options = QFileDialog.Options()
        self.trained_model_path, _ = QFileDialog.getOpenFileName(self, "选择训练好的模型", "", "Model Files (*.pth; *.pkl)")
        if self.trained_model_path:
            print(f"选择的训练好的模型路径: {self.trained_model_path}")
            # 在选择训练好的模型文件后直接加载模型
            self.load_model_from_path(self.trained_model_path)

    def load_model_from_path(self, model_path):
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return

        if model_path.endswith('.pth'):
            if 'cnn' in model_path:
                self.model = CNN().to(device)
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()  # 设置为评估模式
                self.model_type = "cnn"
                print(f"Loaded CNN model from {model_path}")
            elif 'resnet' in model_path:
                self.model = models.resnet18(pretrained=False, num_classes=10).to(device)
                self.model.load_state_dict(torch.load(model_path, map_location=device))
                self.model.eval()  # 设置为评估模式
                self.model_type = "ResNet"
                print(f"Loaded ResNet model from {model_path}")
        elif model_path.endswith('.pkl'):
            if 'decision_tree' in model_path:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_type = "decision_tree"
                print(f"Loaded Decision Tree model from {model_path}")
            elif 'svm' in model_path:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_type = "svm"
                print(f"Loaded SVM model from {model_path}")
            elif 'random_forest' in model_path:
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                self.model_type = "random_forest"
                print(f"Loaded Random Forest model from {model_path}")
            else:
                print("Unsupported model type")
        else:
            print("Unsupported model file type")


    # 连接数据库
    def connect_to_database(self):
        try:
            conn = mysql.connector.connect(
                host='127.0.0.1',
                database='machinelearn',
                user='root',
                password='123456'
            )
            return conn
        except mysql.connector.Error as err:
            print(f"数据库错误: {err}")
            return None

    # 关闭数据库
    def close_database_connection(self, conn, cursor):
        if cursor:
            cursor.close()
        if conn and conn.is_connected():
            conn.close()
            print("数据库连接已关闭。")

    #查询个人用户信息
    def load_personal_info(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT Email, Gender, PasswordHash, RegistrationDate, UserID, Username "
                    "FROM Users "
                    "WHERE UserID = %s",
                    (self.user_id,))
                row = cursor.fetchone()

                if row:
                    email, gender, password_hash, registration_date, user_id, username = row
                    self.ui.lineEdit_Email.setText(email)
                    self.ui.lineEdit_Gender.setText(gender)
                    self.ui.lineEdit_PasswordHash.setText(password_hash)
                    self.ui.lineEdit_RegistrationDate.setText(str(registration_date))
                    self.ui.lineEdit_UserID.setText(str(user_id))
                    self.ui.lineEdit_Username.setText(username)
                else:
                    print("用户未找到。")
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)


    def modify_personal_info(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                user_id = self.ui.lineEdit_UserID.text()
                username = self.ui.lineEdit_Username.text()
                password_hash = self.ui.lineEdit_PasswordHash.text()
                email = self.ui.lineEdit_Email.text()
                gender = self.ui.lineEdit_Gender.text()
                registration_date = self.ui.lineEdit_RegistrationDate.text()

                cursor.execute("""
                    UPDATE Users
                    SET Username = %s, PasswordHash = %s, Email = %s, Gender = %s, RegistrationDate = %s 
                    WHERE UserId = %s
                """, (username, password_hash, email, gender, registration_date, user_id))

                conn.commit()
                print("用户信息更新成功。")

                # 更新界面显示的用户信息
                self.ui.lineEdit_Email.setText(email)
                self.ui.lineEdit_Gender.setText(gender)
                self.ui.lineEdit_PasswordHash.setText(password_hash)
                self.ui.lineEdit_RegistrationDate.setText(registration_date)
                self.ui.lineEdit_Username.setText(username)

            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)


    #查询个人实验信息
    def load_experiments(self):
        conn = self.connect_to_database()
        if conn:
            try:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT UserID,ResultID  ,ModelName, PredictionDate ,PredictionDataPath ,PredictionResult "
                    "FROM Experimentresults "
                    "WHERE UserID = %s",
                    (self.user_id,))
                rows = cursor.fetchall()

                if rows:
                    self.ui.tableWidget.setRowCount(len(rows))
                    self.ui.tableWidget.setColumnCount(6)
                    self.ui.tableWidget.setHorizontalHeaderLabels(['用户ID', '实验ID', '模型名', '实验时间', '预测数据路径', '实验结果'])

                    for row_num, row_data in enumerate(rows):
                        for col_num, data in enumerate(row_data):
                            self.ui.tableWidget.setItem(row_num, col_num, QTableWidgetItem(str(data)))
                else:
                    print("没有找到实验信息。")
            except mysql.connector.Error as err:
                print(f"数据库错误: {err}")
            finally:
                self.close_database_connection(conn, cursor)


    #删除个人实验信息
    def delete_experiment(self):
        selected_row = self.ui.tableWidget.currentRow()
        if selected_row != -1:
            experiment_id = self.ui.tableWidget.item(selected_row, 1).text()  # 获取选中行的实验ID
            conn = self.connect_to_database()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM Experimentresults WHERE ResultID = %s",
                                   (experiment_id,))
                    conn.commit()
                    print(f"实验ID为 {experiment_id} 的实验信息删除成功。")
                    self.load_experiments()  # 重新加载实验信息表格
                except mysql.connector.Error as err:
                    print(f"数据库错误: {err}")
                finally:
                    self.close_database_connection(conn, cursor)
        else:
            QMessageBox.warning(self, "警告", "请先选择要删除的实验信息。", QMessageBox.Ok)


    def look_model(self):
        conn = self.connect_to_database()
        if conn is not None:
            try:
                # 创建游标对象
                cursor = conn.cursor()
                # 执行查询语句，这里需要根据实际情况修改SQL语句
                cursor.execute("SELECT ModelName, ModelEval, DataPath, ModelPath, ModelParameters "
                               "FROM train "
                               "WHERE UserID = %s",
                               (self.user_id,))
                # 获取查询结果
                train_data = cursor.fetchall()

                if train_data:
                    self.ui.tableWidget_4.setRowCount(len(train_data))
                    self.ui.tableWidget_4.setColumnCount(5)
                    self.ui.tableWidget_4.setHorizontalHeaderLabels(
                        ['模型名', '评估效果', '数据路径', '模型路径', '模型参数'])

                    for row_num, row_data in enumerate(train_data):
                        for col_num, data in enumerate(row_data):
                            self.ui.tableWidget_4.setItem(row_num, col_num, QTableWidgetItem(str(data)))

            except mysql.connector.Error as err:
                print(f"数据库查询错误: {err}")
            finally:
                # 关闭游标和数据库连接
                self.close_database_connection(conn, cursor)
        else:
            print("无法连接到数据库。")


    def modify_model(self):
        selected_row = self.ui.tableWidget_4.currentRow()
        if selected_row != -1:
            model_name = self.ui.tableWidget_4.item(selected_row, 0).text()  # 获取选中行的模型名
            conn = self.connect_to_database()
            if conn:
                try:
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM train WHERE ModelName = %s",
                                   (model_name,))
                    conn.commit()
                    print(f"模型名为 {model_name} 的模型信息删除成功。")
                except mysql.connector.Error as err:
                    print(f"数据库错误: {err}")
                finally:
                    self.close_database_connection(conn, cursor)
            else:
                print("无法连接到数据库。")
        else:
            QMessageBox.warning(self, "警告", "请先选择要删除的模型信息。", QMessageBox.Ok)



if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    app = QApplication(sys.argv)
    win = LoginWindow()
    app.processEvents()  # 确保所有挂起的事件都被处理
    sys.exit(app.exec_())
