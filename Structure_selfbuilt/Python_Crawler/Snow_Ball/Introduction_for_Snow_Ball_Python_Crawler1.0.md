# 爬取雪球企业评论数据 📈

本程序使用 **Python + Selenium** 爬取 **雪球(Xueqiu)** 网站上的企业评论数据，并自动保存到 Excel 文件中。

---

## **📈 1. 环境配置**
本程序需要 **Python 3.7+**，并且需要安装以下依赖：
- `selenium`
- `pandas`
- `openpyxl`
- Google Chrome 浏览器（**必须安装**）
- ChromeDriver（**必须匹配 Chrome 版本**）

### **🔹 1.1 安装 Python**
如果你的系统未安装 Python，请先下载并安装：
🔗 [Python 官网下载](https://www.python.org/downloads/)

### **🔹 1.2 安装 Python 依赖**
打开 **终端 (Terminal) / 命令提示符 (CMD)**，执行：
```bash
pip install selenium pandas openpyxl
```

### **🔹 1.3 安装 Google Chrome**
如果你的电脑 **尚未安装 Chrome 浏览器**，请前往：
🔗 [Chrome 官网下载](https://www.google.com/chrome/)

### **🔹 1.4 安装 ChromeDriver**
ChromeDriver 需要 **匹配你的 Chrome 版本**：
1. **检查 Chrome 版本**  
   - 在 Chrome 地址栏输入：
     ```
     chrome://version/
     ```
   - 复制 **版本号**（例如 `110.0.5481.77`）

2. **下载匹配的 ChromeDriver**
   - 访问 [ChromeDriver 官方网站](https://chromedriver.chromium.org/downloads)
   - 下载与你 Chrome 版本匹配的 ChromeDriver
   - 解压 `chromedriver.exe` 并放入 Python 目录（例如 `C:\Python310\Scripts\`）

3. **验证 ChromeDriver 是否安装成功**
   ```bash
   chromedriver --version
   ```

---

## **📈 2. 运行前的准备**
### **🔹 2.1 修改企业名称**
打开 `雪球爬虫.py`，修改 `company_name`：
```python
company_name = "光大证券"  # 这里可以改为任意公司，如 "比亚迪"、"贵州茅台"
```

### **🔹 2.2 设置 ChromeDriver 路径**
修改以下行，确保 `chromedriver_path` 指向 **你的 ChromeDriver 位置**：
```python
chromedriver_path = r"D:\Chrome\chrome_driver\chromedriver-win64\chromedriver-win64\chromedriver.exe"
```
如果你将 `chromedriver.exe` 放在 **系统环境变量** 中，可以省略此步骤。

---

## **📈 3. 如何运行程序**
打开 **终端 (Terminal) / 命令提示符 (CMD)**，切换到代码所在目录：
```bash
cd path/to/your/code
```
然后运行：
```bash
python 雪球爬虫.py
```

程序将会：
1. **打开雪球主页**
2. **搜索企业名称**
3. **自动获取企业股票代码**
4. **爬取该企业的雪球评论**
5. **将数据保存到 Excel 文件**

---

## **📈 4. 输出文件**
数据会自动保存到 Excel，格式如下：
```
雪球_光大证券_评论_2024-03-13_14-30-45.xlsx
```

---

## **📈 5. 可能遇到的问题 & 解决方案**
### **❌ 问题 1：程序运行时报 `chromedriver` 错误**
**🔹 解决方案**
- **确保 ChromeDriver 版本匹配 Chrome**
- **执行 `chromedriver --version` 确认是否可用**
- **检查 `chromedriver_path` 是否正确**

### **❌ 问题 2：无法获取 `API URL`**
**🔹 解决方案**
- **检查是否有雪球反爬虫机制**（可尝试手动访问雪球网站）
- **检查 `company_name` 是否正确**（是否输入了完整公司名称）
- **确保 `网络环境良好`**

### **❌ 问题 3：程序运行后，Excel 文件为空**
**🔹 解决方案**
- **可能是雪球网站限制 API 访问**
- **尝试更换 `User-Agent` 或 `Cookie`**

---

## **📈 6. 计划任务（Windows 定时运行）**
如果你希望 **每天定时运行此爬虫**：
1. **创建一个 `.bat` 文件**
   - 打开 **记事本**
   - 输入以下内容：
     ```bat
     @echo off
     cd /d "C:\Users\你的用户名\Desktop\雪球爬虫"  # 进入爬虫所在目录
     python 雪球爬虫.py  # 运行爬虫
     exit
     ```
   - **保存为 `run_snowball.bat`**

2. **使用 Windows 任务计划程序**
   - 打开 `任务计划程序`
   - 选择 **"创建基本任务"**
   - 设置 **每天定时运行**
   - 选择 `"启动程序"` → 选择 `run_snowball.bat`
   - 保存任务，测试运行

---

## **📈 7. 贡献 & 联系方式**
如果你有任何建议或问题，欢迎提交 Issue 或联系作者。

---

🚀 **试试看，运行 `python 雪球爬虫.py`，看看 Excel 里的数据吧！🎯🔥**

