# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled2.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(501, 340)
        Dialog.setMinimumSize(QtCore.QSize(501, 340))
        Dialog.setMaximumSize(QtCore.QSize(501, 340))
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setTextFormat(QtCore.Qt.RichText)
        self.label.setScaledContents(False)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "关于"))
        self.label.setText(_translate("Dialog", "<html><head/><body><p align=\"center\"><span style=\" font-size:18pt; font-weight:600;\">关于本软件</span><br/></p><p><span style=\" font-size:12pt;\">  “基于神经网络与LINE算法的同名作者消歧义软件”是一</span></p><p><span style=\" font-size:12pt;\">款能有效从文献数据库中大量同名作者之间区分各个单独作者</span></p><p><span style=\" font-size:12pt;\">及他们所写的论文的软件。软件小巧便于操作。</span></p><p><span style=\" font-size:12pt;\">  此软件采用的消歧方法是根据数据库中论文的属性构建二</span></p><p><span style=\" font-size:12pt;\">部图，并基于 LINE方法得到每篇文章的特征向量，之后利用</span></p><p><span style=\" font-size:12pt;\">多层感知机模型区分同名作者。该方法在DBLP数据库中对两百</span></p><p><span style=\" font-size:12pt;\">多位同名作者进行了实验，证明该方法消歧效果优越。</span></p><p><br/></p></body></html>"))
