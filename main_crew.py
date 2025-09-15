from crewai import Crew, Process, Task
from agent_factory import create_repo_analysis_agents
from llm_setup import get_llm

class RepoAnalysisCrew:
    def __init__(self,retriever):
        self.llm= get_llm()
        self.retriever=retriever
        self.agents=create_repo_analysis_agents(self.llm, self.retriever)

    def run(self, repo_path:str, user_query:str):
        # Task Definitions
        # Note: The output of the previous task in a sequential process is implicitly passed as context for later tasks by Crew AI.
        summary_task=Task(
            description=f"""
                Analyze the repository at path '{repo_path}' to generate a high-level summary.
                Focus on the project's main purpose, architecture, and key functionalities.
                The user is particularly interested in: '{user_query}'""",
            expected_output="A concise summary (3-4 paragraphs) of the repository's architecture and purpose",
            agent=self.agents["summarizer"])
        
        dependency_task=Task(
            description=f"Analyze the depenency files in the repository at '{repo_path}'.",
            expected_output="A list of all the project dependencies, categorized by ecosystem (e.g., Python, Node.js).",
            agent=self.agents["dependency_analyst"])
        
        security_task= Task(
            description=f"""
                Scan the codebase at '{repo_path}' for common security vulnerabilities.
                Specifically look for hardcoded secrets (API keys, passwords) and potential SQL injection patterns.""",
            expected_output="A report lisiting any potential securitiy vulnerablities found, with file paths and line numbers if possible.",
            agent=self.agents["security_auditor"] 
        )

        documentation_task= Task(
            description=f"""Combine the findings from the code summary, dependency analysis, and security audit
            into a single, comprehensive report. The report should be well-structured in Markdown format.""",
            expected_output="A final, comprehensive report in Markdown format, with sections for Summary, Dependencies and Security Analysis.",
            agent=self.agents["documentation_generator"],
            context=[summary_task,dependency_task,security_task]
        )

        crew=Crew(
            agents=list(self.agents.values()),
            tasks=[summary_task,dependency_task,security_task,documentation_task],
            process=Process.sequential,
            verbose=True
        )

        print("Kicking off Repo-Intellect crew...")
        result= crew.kickoff()
        print("Crew execution finished.")
        return result