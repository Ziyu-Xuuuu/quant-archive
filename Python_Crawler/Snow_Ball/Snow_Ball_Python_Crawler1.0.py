import time
import json
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import datetime

# **定义要搜索的企业名称**
company_name = "光大证券"  # 修改为你要搜索的公司

# **ChromeDriver 路径**
chromedriver_path = r"D:\Chrome\chrome_driver\chromedriver-win64\chromedriver-win64\chromedriver.exe"

# **Chrome 浏览器选项**
chrome_options = Options()
#chrome_options.add_argument("--headless")  # 无头模式
chrome_options.add_argument("--disable-gpu")  
chrome_options.add_argument("--ignore-certificate-errors")
chrome_options.add_argument("--allow-running-insecure-content")
chrome_options.add_argument("--disable-popup-blocking")
chrome_options.add_argument("--log-level=3")  
chrome_options.add_argument("--silent")  
chrome_options.add_experimental_option("detach", True)
chrome_options.add_argument("--disable-notifications")
chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})  

# **启动 WebDriver**
service = Service(chromedriver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)

try:
    # **1. 打开雪球主页**
    driver.get("https://xueqiu.com/")
    time.sleep(3)

    # **2. 在搜索框输入需检索的企业名称**
    search_box = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.NAME, "q"))
    )
    search_box.send_keys(company_name)
    search_box.submit()
    time.sleep(3)

    # **3. 提取第一个搜索结果的股票代码**
    first_result = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, '//a[contains(@href, "/S/")]'))
    )
    stock_url = first_result.get_attribute("href")
    stock_code = stock_url.split("/")[-1]  # 提取股票代码
    print(f"✅ 识别到 {company_name} 的股票代码: {stock_code}")

    # **4. 点击搜索结果中的第一个股票链接**
    first_result.click()
    time.sleep(3)

    # **5. 监听 Network 请求获取 API URL**
    print("🔍 监听 Network 请求以获取评论 API URL...")
    api_url = None

    logs = driver.get_log("performance")
    for entry in logs:
        try:
            log = json.loads(entry["message"])["message"]
            if log["method"] == "Network.requestWillBeSent":
                url = log["params"]["request"]["url"]
                if "xueqiu.com/query/v1/symbol/search/status" in url:
                    api_url = url.split("&md5__1038")[0]  # 去掉 md5__1038
                    break
        except Exception as e:
            continue

    if not api_url:
        print("❌ 未找到评论 API URL，退出程序！")
        driver.quit()
        exit()

    print(f"✅ 获取到 API URL: {api_url}")

    # **6. 让 Selenium 直接访问 API URL**
    print("🔍 访问雪球 API URL 以获取评论数据...")
    driver.get(api_url)
    time.sleep(3)

    # **7. 获取网页中的 JSON 数据**
    page_source = driver.find_element(By.TAG_NAME, "pre").text  # API 返回 JSON，通常在 `<pre>` 标签中
    data = json.loads(page_source)

    # **8. 解析评论数据**
    comments_data = []
    if "list" in data and isinstance(data["list"], list) and data["list"]:
        comments = data["list"]
        print(f"✅ 成功爬取 {len(comments)} 条评论")

        for comment in comments:
            comment_id = comment["id"]
            user = comment["user"]["screen_name"]
            text = comment["text"]
            like_count = comment.get("like_count", 0)

            comments_data.append([user, text, comment_id, like_count])

    else:
        print("❌ 没有更多评论，爬取结束！")

except Exception as e:
    print("❌ 发生错误:", e)

finally:
    driver.quit()

# **9. 保存数据到 Excel**
# **获取当前日期和时间**
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
df = pd.DataFrame(comments_data, columns=["用户名", "评论内容", "评论 ID", "点赞数"])
excel_filename = f"雪球_{company_name}_评论_{current_time}.xlsx"
df.to_excel(excel_filename, index=False, engine="openpyxl")

print(f"✅ 所有评论已保存到 {excel_filename}，共 {len(comments_data)} 条")
