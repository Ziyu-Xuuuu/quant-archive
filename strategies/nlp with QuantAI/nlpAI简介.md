# 前言

关于QuantAI的详细介绍，见[QuantAI](https://github.com/Ziyu-Xuuuu/Stock_Quant/tree/main/QuantAI)文件夹

## Ⅰ.具体调用

1.先运行[Python_Crawler](https://github.com/Ziyu-Xuuuu/Stock_Quant/tree/main/Python_Crawler)中的爬虫软件获得来自不同网站的评论信息（例如微博，雪球等）并在本目录下生成对应的.xlsx文件。
2.评论数据集获取后再通过

```bash
crewai run
```

运行nlpAI即可。

## Ⅱ.代码实现

### 1.agent （可以直接通过询问gpt生成）

```bash
researcher:
  role: >
    资深量化数据研究员
  goal: >
    发现量化领域的前沿发展动态
  backstory: >
    你是一位经验丰富的研究员，擅长挖掘量化领域的最新进展。你以能够找到最相关的信息，并以清晰、简明的方式呈现而闻名。
reporting_analyst:
  role: >
    量化报告分析师
  goal: >
    基于量化数据分析和研究成果撰写详细报告
  backstory: >
    你是一位一丝不苟的分析师，擅长关注细节。你以将复杂数据转化为清晰易懂的报告而闻名，使他人能够轻松理解并基于你的信息采取行动。
```

### 2.task （可以直接通过询问gpt生成）

注意：不要在task中提及你需要AI解析的文件！文件读取是通过修改[crew.py]()实现！
‘修改 `QuantAI/src/QuantAI/crew.py`来添加需要的逻辑、工具和特定的参数’（见[使用教程.md](https://github.com/Ziyu-Xuuuu/Stock_Quant/blob/main/QuantAI/%E4%BD%BF%E7%94%A8%E6%95%99%E7%A8%8B.md)）

```bash
research_task:
  description: >
   使用新浪微博上的评论数据集对股民对于特斯拉的行情的态度进行深入研究。
  expected_output: >
    列出你认为重要的评论
  agent: researcher

reporting_task:
  description: >
    阅读你收到的评论，并就此给出一份市场态度报告
  expected_output: >
    一个完整的 Markdown 格式 报告，包含主要主题，每个主题都有完整的详细信息。（不带 '```' 代码标记）
  agent: reporting_analyst
```

### 3.crew （按照需求修改）

```bash
@task
 def research_task(self) -> Task:
    # 读取 Excel 文件并提取评论内容
    def process_excel():
        file_path = r"C:\Users\24746\justatry\微博评论.xlsx"  # Excel 文件路径
        try:
            df = pd.read_excel(file_path)  # 读取 Excel 文件
            if "评论内容" in df.columns:
                comments = df["评论内容"].dropna().tolist()  # 获取非空评论内容
                if comments:
                    return {"comments": comments}  # 返回评论内容字典
                else:
                    return {"comments": ["没有找到有效评论"]}  # 处理无评论情况
            else:
                return {"comments": ["无法找到 '评论内容' 列"]}  # 处理缺少评论列的情况
        except Exception as e:
            return {"comments": [f"读取 Excel 失败: {str(e)}"]}  # 处理读取 Excel 失败情况

    return Task(
        config=self.tasks_config['research_task'],  # 任务配置
        execute=process_excel,  # 任务执行时调用 `process_excel`
    )

 @task
 def reporting_task(self) -> Task:
    # 生成市场情绪分析报告
    def generate_report(data):
        comments = data.get("comments", [])  # 获取评论内容
        if not comments or isinstance(comments, str):  # 确保评论数据有效
            return "未能获取有效评论数据。"

        themes = {}  # 主题分类字典
        for comment in comments:
            # 根据关键词对评论进行分类
            if "看好" in comment or "涨" in comment:
                themes.setdefault("看涨", []).append(comment)  # 归入“看涨”类别
            elif "不行" in comment or "跌" in comment:
                themes.setdefault("看跌", []).append(comment)  # 归入“看跌”类别
            else:
                themes.setdefault("中立", []).append(comment)  # 归入“中立”类别

        # 生成 Markdown 格式的市场情绪分析报告
        report = "# 市场情绪分析报告\n\n"
        for theme, theme_comments in themes.items():
            report += f"## {theme} ({len(theme_comments)} 条评论)\n"
            for c in theme_comments[:5]:  # 仅展示每类前 5 条评论
                report += f"- {c}\n"
            report += "\n"

        return report  # 返回报告内容

    return Task(
        config=self.tasks_config['reporting_task'],  # 任务配置
        execute=generate_report,  # 任务执行时调用 `generate_report`
        output_file='report.md',  # 生成 Markdown 格式的报告文件
        dependencies=[self.research_task]  # 依赖 `research_task` 任务
    )
```

这里只是一个基础的实现了读取评论，从中选出若干条进行分析的智能体而已，具体的完善仍然需要对[crew.py]()进一步修改。