import pandas as pd
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class nlp_with_QuantAI():
    """nlp_with_QuantAIcrew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            verbose=True
        )

    @agent
    def reporting_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['reporting_analyst'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        def process_excel():
            file_path = r"C:\Users\24746\anaconda3\envs\stock_quant_env\Stock_Quant\strategies\nlp_with_QuantAI\微博评论.xlsx"
            try:
                df = pd.read_excel(file_path)  # 读取 Excel 文件
                if "评论内容" in df.columns:
                    comments = df["评论内容"].dropna().tolist()  # 提取非空评论
                    if comments:
                        return {"comments": comments}  # 传递字典，CrewAI 任务可以接收
                    else:
                        return {"comments": ["没有找到有效评论"]}
                else:
                    return {"comments": ["无法找到 '评论内容' 列"]}
            except Exception as e:
                return {"comments": [f"读取 Excel 失败: {str(e)}"]}

        return Task(
            config=self.tasks_config['research_task'],
            execute=process_excel,  # 任务执行时调用 `process_excel`
        )

    @task
    def reporting_task(self) -> Task:
        def generate_report(data):
            comments = data.get("comments", [])
            if not comments or isinstance(comments, str):  
                return "未能获取有效评论数据。"

            themes = {}  # 主题分类
            for comment in comments:
                # 简单关键词归类（可以优化为 NLP 处理）
                if "看好" in comment or "涨" in comment:
                    themes.setdefault("看涨", []).append(comment)
                elif "不行" in comment or "跌" in comment:
                    themes.setdefault("看跌", []).append(comment)
                else:
                    themes.setdefault("中立", []).append(comment)

            # 生成 Markdown 报告
            report = "# 市场情绪分析报告\n\n"
            for theme, theme_comments in themes.items():
                report += f"## {theme} ({len(theme_comments)} 条评论)\n"
                for c in theme_comments[:5]:  # 只展示前 5 条
                    report += f"- {c}\n"
                report += "\n"

            return report

        return Task(
            config=self.tasks_config['reporting_task'],
            execute=generate_report,  # 任务执行时调用 `generate_report`
            output_file='report.md',
            dependencies=[self.research_task]  # 依赖 `research_task`
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
