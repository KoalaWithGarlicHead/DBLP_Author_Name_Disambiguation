import warnings

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import sys
import os
import threading
import re
from uipy.mainWindow import Ui_MainWindow
from uipy.dialog1 import Ui_Dialog as Ui_Dialog1
from uipy.dialog2 import Ui_Dialog as Ui_Dialog2
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QDialog
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot
from baselineOfRandomWalk.integration import disambiguation

import scipy.special.cython_special

resultFileName = "result"


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

        self.openFileAction.triggered.connect(self.openFile)
        self.openInstructionAction.triggered.connect(self.openInstruction)
        self.openAboutAction.triggered.connect(self.openAbout)

        self.pushButton.clicked.connect(self.openFile)
        self.pushButton_2.clicked.connect(self.openDirectory)
        self.pushButton_3.clicked.connect(self.start)
        self.pushButton_4.clicked.connect(self.cancel)

        self.progressBar.setVisible(False)
        self.progressBar.setValue(0)

        self.label_5.setText("")

        self.win2 = MyWindow2()
        self.win3 = MyWindow3()

        self.resource = None
        self.target = None
        self.authorName = None
        self.dataIndex = 0
        self.data = ""
        self.result = None
        self.status = 0
        # 0 for idle
        # 1 for disambiguating
        # 2 for finished
        # -1 for stopped

        self.newThread = None

    def openFile(self):
        file, ok = QFileDialog.getOpenFileName(self, "打开", "C:/", "Text Files(*.txt)")
        if file:
            self.resource = open(file, "r", encoding="utf-8")
            self.data = self.resource.read()
            self.lineEdit.setText(file)
            self.textBrowser.setText(self.data)

    def openDirectory(self):
        directory = QFileDialog.getExistingDirectory(self, "选择文件夹", "C:/")
        if directory:
            file = directory + "/" + resultFileName + ".txt"
            if os.path.exists(file):
                reply = QMessageBox.question(self, '询问', "该位置已存在result.txt文件，是否覆盖？",
                                             QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
                if reply != QMessageBox.Yes:
                    return
            self.target = open(file, "w", encoding="utf-8")
            self.lineEdit_2.setText(file)

    def openInstruction(self):
        self.win2.show()

    def openAbout(self):
        self.win3.show()

    def start(self):
        self.authorName = self.lineEdit_3.text()
        self.dataIndex = self.comboBox.currentIndex()
        if not (self.resource and self.target):
            QMessageBox.information(self, "Error", "请选择源文件和目标文件")
        elif not self.authorName:
            QMessageBox.information(self, "Error", "请输入消歧作者名")
        else:
            if self.status == 1:
                reply = QMessageBox.question(self, '询问', "确定要重新开始消歧吗？（正在进行中的消歧将被终止）",
                                             QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
                if reply == QMessageBox.Yes:
                    self.progressBar.setValue(0)
                    self.startThread()
            else:
                self.status = 1
                self.label_5.setText("正在消歧中...")
                self.progressBar.setVisible(True)
                self.progressBar.setValue(0)
                self.startThread()

    def cancel(self):
        if self.status == 1:
            reply = QMessageBox.question(self, '询问', "确定要终止消歧吗？（终止后将无法恢复）", QMessageBox.Yes | QMessageBox.Cancel,
                                         QMessageBox.Cancel)
            if reply == QMessageBox.Yes:
                self.status = -1
                self.label_5.setText("消歧已终止...")
                self.newThread.terminate()

    def finish(self):
        if self.result and self.status != -1:
            self.progressBar.setValue(100)
            self.status = 2
            self.label_5.setText("消歧完成！")
            QMessageBox.information(self, "提示", "消歧完成")
            self.write()
        else:
            self.errorOccur()

    def errorOccur(self):
        self.status = -1
        self.label_5.setText("消歧已终止...")
        QMessageBox.information(self, "Error", "消歧过程中发生错误")

    def startThread(self):
        if self.newThread:
            self.newThread.terminate()
        self.newThread = myThread(self)
        self.newThread.trigger.connect(self.finish)
        self.newThread.start()

    def write(self):
        originalData = self.data.split('\n\n')
        elementsID = []
        elementsTitle = []
        for i in range(0, len(originalData)):
            if originalData[i] != '':
                elementsID.append(re.search('id:[0-9]+\n', originalData[i]).group()[3:-1])
                elementsTitle.append(re.search('title:[^\n]+', originalData[i]).group()[6:])
        rstList = list(enumerate(self.result))
        rstList.sort(key=lambda x: (x[1], x[0]))
        num = 0
        rstToWrite = []
        for e in rstList:
            if e[1] != num:
                rstToWrite.append("===================================\n" + self.authorName + str(e[1]) + ":\n\n")
                num = e[1]
            rstToWrite.append("ID:" + elementsID[e[0]] + "\n")
            rstToWrite.append("Title:" + elementsTitle[e[0]] + "\n\n")
        rstToWrite = "基于word2vec与聚类模型的消歧结果如下：\n" + ''.join(rstToWrite)
        if self.target:
            self.target.write(rstToWrite)
            self.target.flush()
            self.textBrowser_2.setText(rstToWrite)
            self.label_5.setText("消歧完成！结果已写入目标文件。")

    def end(self):
        self.resource.close()
        self.target.close()


class MyWindow2(QDialog, Ui_Dialog1):
    def __init__(self, parent=None):
        super(MyWindow2, self).__init__(parent)
        self.setupUi(self)


class MyWindow3(QDialog, Ui_Dialog2):
    def __init__(self, parent=None):
        super(MyWindow3, self).__init__(parent)
        self.setupUi(self)


class myThread(QThread):
    trigger = pyqtSignal()

    def __init__(self, win):
        super(myThread, self).__init__()
        self.win = win

    def run(self):
        try:
            cor = disambiguation(self.win.data, self.win.authorName, 20, word2vecSize=20, word2vecWindow=3,
                                 word2vecP=0.75)
            while True:
                rate = next(cor)
                if rate == -1:
                    print("done")
                    self.win.result = next(cor)
                    break
                self.win.progressBar.setValue(rate)
        except:
            self.status = -1
        self.trigger.emit()


if __name__ == '__main__':
    if not QtWidgets.QApplication.instance():
        app = QtWidgets.QApplication(sys.argv)
    else:
        app = QtWidgets.QApplication.instance()
    myWin = MyWindow()
    myWin.show()
    e = app.exec_()  # 主循环，返回状态码
    myWin.end()
sys.exit(e)
