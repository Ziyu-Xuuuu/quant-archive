import requests
import pandas as pd
import time
import json
import html
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# **雪球 API 相关配置**
symbol = "SH601788"
base_url = "https://xueqiu.com/query/v1/symbol/search/status.json"

# **初始化 Selenium**
chrome_options = Options()
chrome_options.add_argument("--headless")  # 无头模式
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

# **启动 ChromeDriver**
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# **打开雪球网页，获取 Cookie**
driver.get(f"https://xueqiu.com/S/{symbol}")
time.sleep(3)  # 等待加载

# **获取最新的 Cookie**
cookies = {c["name"]: c["value"] for c in driver.get_cookies()}
cookie_str = "; ".join([f"{key}={value}" for key, value in cookies.items()])

# **关闭浏览器**
driver.quit()

# **请求头**
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Referer": f"https://xueqiu.com/S/{symbol}",
    "X-Requested-With": "XMLHttpRequest",
    "Cookie": cookie_str,  # **使用 Selenium 获取的 Cookie**
}

# **存储评论数据**
comments_data = []
page = 1
max_pages = 10  # **最多爬取页数**

print(f"🔥 正在爬取雪球 {symbol} 的评论数据...")

while page <= max_pages:
    # **动态获取 md5__1038**
    driver = webdriver.Chrome(service=service, options=chrome_options)
    driver.get(f"https://xueqiu.com/S/{symbol}")
    time.sleep(3)

    logs = driver.get_log("performance")
    driver.quit()

    # **解析 `XHR` 请求，获取 `md5__1038`**
    md5_1038 = None
    for log in logs:
        log_entry = json.loads(log["message"])["message"]
        if "params" in log_entry and "request" in log_entry["params"]:
            url = log_entry["params"]["request"].get("url", "")
            if "md5__1038=" in url:
                md5_1038 = url.split("md5__1038=")[-1].split("&")[0]
                break

    if not md5_1038:
        print("❌ 未找到 `md5__1038` 参数，跳过当前页！")
        break

    # **构造 API 请求**
    params = {
        "count": 10,
        "comment": 0,
        "symbol": symbol,
        "hl": 0,
        "source": "all",
        "sort": "",
        "page": page,
        "q": "",
        "type": 11,
        "md5__1038": md5_1038,
    }

    try:
        response = requests.get(base_url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"❌ 返回内容: {response.text[:200]}")
            break

        data = response.json()

        if "list" in data and isinstance(data["list"], list) and data["list"]:
            comments = data["list"]
            print(f"✅ 第 {page} 页，爬取 {len(comments)} 条评论")

            for comment in comments:
                comment_id = comment["id"]
                user = comment["user"]["screen_name"]
                text = html.unescape(comment["text"])
                like_count = comment.get("like_count", 0)
                reply_count = comment.get("reply_count", 0)

                comments_data.append([user, text, "否", comment_id, like_count])

                # **爬取子评论**
                if reply_count > 0:
                    print(f"🔍 发现 {reply_count} 条子评论，正在爬取...")
                    sub_params = {
                        "symbol_id": symbol,
                        "id": comment_id,
                        "count": 10,
                        "type": "comment",
                    }
                    sub_response = requests.get(base_url, headers=headers, params=sub_params)

                    if sub_response.status_code == 200:
                        try:
                            sub_data = sub_response.json()
                            if "list" in sub_data and isinstance(sub_data["list"], list):
                                for sub_comment in sub_data["list"]:
                                    sub_user = sub_comment["user"]["screen_name"]
                                    sub_text = html.unescape(sub_comment["text"])
                                    sub_like_count = sub_comment.get("like_count", 0)
                                    comments_data.append([sub_user, sub_text, "是", comment_id, sub_like_count])
                        except Exception as e:
                            print(f"❌ 解析子评论 JSON 失败: {e}")
                            print(f"❌ 返回数据: {sub_response.text[:200]}")
                    else:
                        print(f"❌ 子评论请求失败，状态码: {sub_response.status_code}")
                        print(f"❌ 返回内容: {sub_response.text[:200]}")

            page += 1  # **翻页**
            time.sleep(2)  # **防止被封**
        else:
            print("❌ 没有更多评论，爬取结束！")
            break

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络错误: {e}")
        break
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        break

# **🔥 解决 `FutureWarning`**
df = pd.DataFrame(comments_data, columns=["用户名", "评论内容", "是否为子评论", "父评论 ID", "点赞数"])
df = df.map(lambda x: x.encode('utf-8', 'ignore').decode('utf-8') if isinstance(x, str) else x)

# **✅ 保存到 Excel**
excel_filename = "雪球评论.xlsx"
with pd.ExcelWriter(excel_filename, engine="openpyxl") as writer:
    df.to_excel(writer, index=False)

print(f"✅ 所有评论已保存到 {excel_filename}，共 {len(comments_data)} 条")
