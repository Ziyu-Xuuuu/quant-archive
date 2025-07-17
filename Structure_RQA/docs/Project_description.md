# 仓库使用方法
**整个项目目前正在使用了RQA平台提供的回测和研究，模拟交易等功能来实现量化，包括两个structures，其中RQA为正在使用的structure，self-built为暂不使用的structure**
**RQA详见https://rqalpha.readthedocs.io/zh-cn/latest/index.html**
## 整体结构
1. src: 包含RQA的github repo，不参加项目的运行
2. baseline： 包含test代码，用来验证仓库是否配置正常
3. docs：文档，包括项目说明文档和个人研究想法记录
## 安装指导
1. 第一步先克隆仓库 
``` 
git clone https://github.com/Ziyu-Xuuuu/Stock_Quant.git
```
2. 安装RQA平台
```
pip install -i https://pypi.douban.com/simple rqalpha
```
3. 验证平台是否正常工作,如果有输出，则证明工作正常
```
rqalpha version
```
4. 下载股票数据
```
rqalpha download-bundle
```
5. 使用baseline进行测试,你会看到回测的图象（这里需要将文件路径改为绝对路径）
```
rqalpha run -f /home/ziyu/Desktop/Stock_Quant/Structure_RQA/baseline/test.py -s 2016-06-01 -e 2016-12-01 --account stock 100000 --benchmark 000300.XSHG --plot
```
