import requests
import pandas as pd
import time
import html

# 微博 API URL
base_url = "https://weibo.com/ajax/statuses/buildComments"

# 目标微博 ID
weibo_id = "5138188316967325"  # ⚠️你要爬取的微博 ID
max_id = 0  # 翻页参数
max_id_type = 0  # 翻页类型

# 请求头（更新 Cookie！）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Referer": "https://weibo.com",
    "Accept": "application/json, text/plain, */*",
    # ⚠️需要手动更新新的cookie
    "Cookie": "XSRF-TOKEN=MaNArS-H7XJ0T0PEz5VVvbhO; SCF=ArbuT8p6xqek-Q-M_fWvqwcVVMceO_sjP2cMnaLGtQXtljAqgcB3BGfQ6m3q8GXzR5IKrMTMcNFSE8fBYRTo__A.; SUB=_2A25KxDVoDeRhGeFH41YS-SnKzTqIHXVpuMigrDV8PUNbmtAbLWKhkW9NeitDsXG-ac7UFGLhkdqi3mmG9MzcIRx6; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9WFzqjHWa2CwaFg38WsmCio45NHD95QN1KnXe0.NSoqcWs4DqcjMi--NiK.Xi-2Ri--ciKnRi-zNS0.RShe4S0qcSntt; ALF=02_1743245880; WBPSESS=Mxxcq4B5zOJHDRWiRwxm7tPPyp5j3ldCbM1bMWAYkfuESFJchHeKkfhEHNRL2eTVP0Lolp7x9OvWXuYwW19oXylMYFflPe3T9cXfPEOKX-LAVMqavy4VcF-t2hPksD5JqIQ1VAsMid3MKoCZPZJhXA==",
}
# 存储评论数据
comments_data = []

print(f"🔥 正在爬取微博 {weibo_id} 的评论数据...")

# 设定分页参数
max_id = 0
max_id_type = 0

while True:  # 继续翻页，直到 max_id=0
    params = {
        "is_reload": "1",
        "id": weibo_id,
        "count": "20",  # 一次最多返回 20 条
        "max_id": max_id,  # 关键参数，控制翻页
        "max_id_type": max_id_type,
        "fetch_level": "0",
        "locale": "zh-CN",
        "uid": "6004281123",
        "is_show_bulletin": "2",
        "is_mix": "0"
    }

    try:
        response = requests.get(base_url, headers=headers, params=params)
        data = response.json()

        if "data" in data and data["data"]:
            comments = data["data"]

            for comment in comments:
                comment_id = comment["id"]
                user = comment["user"]["screen_name"]
                text = html.unescape(comment["text"])
                like_count = comment.get("like_counts", 0)

                # 存储主评论
                comments_data.append([user, text, "否", comment_id, like_count])

                # 直接解析子评论
                if "comments" in comment and comment["comments"]:
                    for sub_comment in comment["comments"]:
                        sub_user = sub_comment["user"]["screen_name"]
                        sub_text = html.unescape(sub_comment["text"])
                        sub_like_count = sub_comment.get("like_counts", 0)

                        comments_data.append([sub_user, sub_text, "是", comment_id, sub_like_count])

            print(f"✅ 爬取 {len(comments)} 条评论（包括子评论）")

            # **更新 max_id 进行翻页**
            max_id = data.get("max_id", 0)  # 获取下一页的 max_id
            max_id_type = data.get("max_id_type", 0)  # 获取下一页的 max_id_type

            if max_id == 0:
                print("🚀 所有评论爬取完成！")
                break  # 如果 max_id=0，则说明没有更多评论，退出循环

            time.sleep(1)  # 避免请求过快被封

        else:
            print("❌ 没有更多评论，爬取结束！")
            break

    except requests.exceptions.RequestException as e:
        print(f"❌ 网络错误: {e}")
        break
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        break

# 保存到 Excel
excel_filename = "微博评论.xlsx"
df = pd.DataFrame(comments_data, columns=["用户名", "评论内容", "是否为子评论", "父评论 ID", "点赞数"])
df.to_excel(excel_filename, index=False, engine="openpyxl")

print(f"✅ 所有评论已保存到 {excel_filename}，共 {len(comments_data)} 条")