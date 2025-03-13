from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
import os
# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class LatestAiDevelopment():
	"""LatestAiDevelopment crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'
	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
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

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def research_task(self) -> Task:
		def process_python_file():
			file_path = r"D:\Stock_Quant\Stock_Quant\strategies\bollinger_strategy.py"
    
			try:
				if not os.path.exists(file_path):
					return {"code": ["文件不存在"]}

				with open(file_path, "r", encoding="utf-8") as file:
					code_content = file.readlines()  # 逐行读取代码
        
				if code_content:
					return {"code": code_content}  # 以字典的方式返回代码
				else:
					return {"code": ["文件为空"]}
    
			except Exception as e:
				return {"code": [f"读取 Python 文件失败: {str(e)}"]}  
		return Task(
			config=self.tasks_config['research_task'],
			execute=process_python_file,  # 任务执行时调用 `process_python_file`		
		)

	@task
	def reporting_task(self) -> Task:
		return Task(
			config=self.tasks_config['reporting_task'],
			output_file='report.md'
		)

	@crew
	def crew(self) -> Crew:
		"""Creates the LatestAiDevelopment crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
