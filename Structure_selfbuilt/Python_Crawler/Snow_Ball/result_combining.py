import os
import re
import sys
from datetime import datetime
import pandas as pd

# 正则表达式提取日期的模式
pattern = re.compile(r".*_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})\.xlsx")

# 检查命令行参数
if len(sys.argv) != 3:
    print("用法：python merge_recent_excels.py 开始日期(yyyy-mm-dd) 结束日期(yyyy-mm-dd)")
    sys.exit()

# 获取日期范围
start_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
end_date = datetime.strptime(sys.argv[2], "%Y-%m-%d").date()

# 存储符合条件的文件及其时间戳
excel_files = []

# 遍历当前目录所有文件
for filename in os.listdir('.'):
    match = pattern.match(filename)
    if match:
        file_date = datetime.strptime(match.group(1), "%Y-%m-%d").date()
        file_time = datetime.strptime(match.group(2), "%H-%M-%S").time()
        if start_date <= file_date <= end_date:
            excel_files.append((file_date, file_time, filename))

# 如果没有符合条件的文件
if not excel_files:
    print("没有符合条件的Excel文件。"); exit()

# 按日期时间倒序排序
excel_files.sort(reverse=True)

# 读取并合并数据，并添加文件日期时间辅助排序
combined_df = pd.DataFrame()
for file_date, file_time, file in excel_files:
    df = pd.read_excel(file)
    df['_file_datetime'] = datetime.combine(file_date, file_time)
    combined_df = pd.concat([combined_df, df], ignore_index=True)

# 按文件日期时间降序排序
combined_df.sort_values(by='_file_datetime', ascending=False, inplace=True)

# 删除重复行（不考虑'点赞数'列，仅保留最新记录）
cols_to_check = [col for col in combined_df.columns if col not in ['点赞数', '_file_datetime']]
combined_df.drop_duplicates(subset=cols_to_check, keep='first', inplace=True)

# 删除辅助列
combined_df.drop(columns=['_file_datetime'], inplace=True)

# 保存合并后的数据到Excel
output_filename = f'合并结果_{start_date}_到_{end_date}.xlsx'
combined_df.to_excel(output_filename, index=False)

print(f'已合并Excel文件，结果保存在：{output_filename}')