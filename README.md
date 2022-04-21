# DBLP数据库作者姓名消歧小工具

## 运行环境

python == 3.6

| Package             | Version   |
| ------------------- | --------- |
| absl-py             | 1.0.0     |
| astor               | 0.8.1     |
| ca-certificates     | 2022.3.29 |
| cached-property     | 1.5.2     |
| certifi             | 2020.6.20 |
| click               | 8.0.4     |
| cycler              | 0.11.0    |
| dataclasses         | 0.8       |
| decorator           | 5.1.1     |
| fastdtw             | 0.3.2     |
| gast                | 0.5.3     |
| gensim              | 3.6.0     |
| grpcio              | 1.45.0    |
| h5py                | 3.1.0     |
| importlib-metadata  | 4.8.3     |
| importlib-resources | 5.4.0     |
| joblib              | 0.13.0    |
| keras-applications  | 1.0.8     |
| keras-preprocessing | 1.1.2     |
| kiwisolver          | 1.3.1     |
| libcxx              | 12.0.0    |
| libffi              | 3.3       |
| markdown            | 3.3.6     |
| matplotlib          | 3.3.4     |
| ncurses             | 6.3       |
| networkx            | 2.1       |
| nltk                | 3.6.7     |
| numpy               | 1.19.5    |
| openssl             | 1.1.1n    |
| pandas              | 1.1.5     |
| pillow              | 8.4.0     |
| pip                 | 21.2.2    |
| protobuf            | 3.19.4    |
| pyparsing           | 3.0.8     |
| pyqt5               | 5.15.6    |
| pyqt5-qt5           | 5.15.2    |
| pyqt5-sip           | 12.9.1    |
| python              | 3.6.13    |
| python-dateutil     | 2.8.2     |
| pytz                | 2022.1    |
| readline            | 8.1.2     |
| regex               | 2022.3.15 |
| scikit-learn        | 0.24.2    |
| scipy               | 1.5.4     |
| setuptools          | 58.0.4    |
| six                 | 1.16.0    |
| smart-open          | 5.2.1     |
| sqlite              | 3.38.2    |
| tensorboard         | 1.12.2    |
| tensorflow          | 1.12.0    |
| termcolor           | 1.1.0     |
| threadpoolctl       | 3.1.0     |
| tk                  | 8.6.11    |
| torch               | 1.10.2    |
| tqdm                | 4.64.0    |
| typing-extensions   | 4.1.1     |
| werkzeug            | 2.0.3     |
| wheel               | 0.37.1    |
| xz                  | 5.2.5     |
| zipp                | 3.6.0     |
| zlib                | 1.2.12    |



## 代码说明

### `code001`

word2vec模型+聚类方法进行消歧

运行app.py即可运行，在源文件处选择`name-unmarked-clean.txt`文件即可

### `code002`

LINE+神经网络进行消歧

运行app.py即可运行，在源文件处选择`name-unmarked-clean.txt`, 标签文件选择`name-rst-clean.txt`文件即可。系统将自动选择75%的数据为训练数据，25%的数据为测试数据，输出文件为消歧的准确率，召回率与F1。

### `code003`

word2vec+神经网络进行消歧

运行app.py即可运行，在源文件处选择`name-unmarked-clean.txt`, 标签文件选择`name-rst-clean.txt`文件即可。系统将自动选择75%的数据为训练数据，25%的数据为测试数据，输出文件为消歧的准确率，召回率与F1。



## 数据集说明

DBLP_People_Data文件，包含了已经经过数据清洗及预处理的DBLP作者数据。