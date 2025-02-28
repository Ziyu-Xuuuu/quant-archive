import requests
import pandas as pd
import time
import html

# ⚠️**雪球 API**
base_url = "https://xueqiu.com/query/v1/symbol/search/status.json?count=10&comment=0&symbol=SH601788&hl=0&source=all&sort=&page=1&q=&type=11&md5__1038=eqRx0ie7qYq2G8DlOmq0%3DD8Q3B4zxR7oD"

# **股票代码**
symbol = "SH601788"

# **请求头（⚠️ 需要替换 Cookie）**
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
    "Referer": f"https://xueqiu.com/S/{symbol}",
    "X-Requested-With": "XMLHttpRequest",  # **🔍 可能必需**
    "Cookie": "cookiesu=281740657651585; Hm_lvt_1db88642e346389874251b5a1eded6e3=1740657653; HMACCOUNT=0FB22BBD9656CD3F; device_id=5d6fb836233790b94d2268189437d253; smidV2=202502272000538efa9d0f545037d9b707602e55da4c5b007ddfc2f9590efb0; s=ak13ze528s; xq_a_token=1c67c1e8a35053705746a994103aaef205409088; xqat=1c67c1e8a35053705746a994103aaef205409088; xq_r_token=5471a5024569b90587b85e7c0804cf16c2366f83; xq_id_token=eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJ1aWQiOjQ3OTM0MDkzMTAsImlzcyI6InVjIiwiZXhwIjoxNzQzMjQ4MDMxLCJjdG0iOjE3NDA2NTc3NDc3MDQsImNpZCI6ImQ5ZDBuNEFadXAifQ.UzPfIoYMDwC0CpcubrAK5C2WiPZYjgizfAjh7iu1gd_7qXcu3rARkeNYxA4LDy3plheLsHy766w2T6UqgKhk-Tl3Ae04mUiTsChNhvJg11LU2N1d_i3cdWdZvitK351qRKe8jsZhRu-v_l_edfMkOrxICZ1MurxucBHGuiVKkxp4TtUJKI8Fj_C52WF0D18rdWhaayER6u4uZTgqOYs7Sy2BFC3MIRjVjYyqxXcaWHDxXCnFQeZsekrCVRyBXzctpYM5SOSeqd63wr5UqJD4m-m4-sDkYq8HYsUcxz6oS0Qd0f7M6ijXRzCB1sSIJnZfZGrJykxSov9bn1zlvFr10A; xq_is_login=1; u=4793409310; .thumbcache_f24b8bbe5a5934237bbc0eda20c1b6e7=z9mGEkePPEzOdxkC5sQcSkBrg834PocDb44QjMTif+s+M3Pz2F0eUw8gZZwlQqk22W1QD2sO4bWRnKg8e5TX4w%3D%3D; acw_tc=ac11000117406594564851800e00389ec092c009a4b55f083cd87ea4d9fa22; is_overseas=0; ssxmod_itna=euD=DI4fxjrhODBD4kDRraDyjxIazDKq0dGMD3qiQGgDYq7=GFDmx0Pkp92xUq5d5QzWYRU1DBkG+rDnqD82DQeDvKPwBh1QnCixxdovE2RG1Ccmn58RQt8Ol2q2TRPlDsFK=keGLDmKDUZ0zqt4Dx=PD5xDTDWeDGDD3TxGaDmeDeOEhD0RwaIgINIBoD7eDXxGChCRwDYPDWxDFN+Dq5lB+TDDCDiyEMt3DixiabzDDB10R5QTe=zDDElYI/pSohnMq7b0wD7y3Dlp4hwMU+T/xz+g9eeZrozSEDCKDjcfpDKb=b4fIKK=ZWGwQw4Q+jBjrAGmlGxexdGG50Dqzv57hYBGOAeZQG5ADQUQU7mjWIjDGIt73u4Y9TlAyZzW/9yY3bQK0CmTt52qn0QUaq+07Q4Njp4Dbq90bQ3i90QW0zKGDD; ssxmod_itna2=euD=DI4fxjrhODBD4kDRraDyjxIazDKq0dGMD3qiQGgDYq7=GFDmx0Pkp92xUq5d5QzWYRUrD88nhr3O9CgRxMSZdBKPNuziD; Hm_lpvt_1db88642e346389874251b5a1eded6e3=1740659658",  # **⚠️ 替换为你的最新 Cookie**
}


# **存储评论数据**
comments_data = []
max_id = None  # **🔍 翻页参数**
page = 1  # **初始页**
max_pages = 10  # **最多爬取页数**

print(f"🔥 正在爬取雪球 {symbol} 的评论数据...")

while page <= max_pages:
    params = {
        "symbol_id": symbol,
        "count": 20,  # 每页 20 条评论
        "max_id": max_id if max_id else "",  # **翻页 ID**
        "type": "status",
    }

    try:
        response = requests.get(base_url, headers=headers, params=params)

        # **🔍 检查 HTTP 状态码**
        if response.status_code != 200:
            print(f"❌ 请求失败，状态码: {response.status_code}")
            print(f"❌ 返回内容: {response.text[:200]}")
            break

        data = response.json()

        # **🔍 确保 `data["list"]` 存在**
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

                # **🔥 爬取子评论**
                if reply_count > 0:
                    print(f"🔍 发现 {reply_count} 条子评论，正在爬取...")
                    sub_params = {
                        "symbol_id": symbol,
                        "id": comment_id,
                        "count": 10,  # 每页 10 条子评论
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

            # **更新 `max_id` 进行翻页**
            max_id = data.get("next_max_id")
            if not max_id:
                print("🚀 没有更多评论，爬取结束！")
                break

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