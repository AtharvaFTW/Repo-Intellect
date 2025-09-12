import os
from crewai import Agent
from crewai.tools import BaseTool
from llm_setup import get_llm

class DependencyAnalysisTool(BaseTool):
    name:str="Dependency Anaylsis Tool"
    description:str="Parses common dependency files like requirements.txt or package.json to a list of project dependencies."

    def run(self,directory:str)->str:
        dependency_files= {
            "requirements.txt":"Python (pip)",
            "package.json":"Node.js(npm)"
        }

        found_dependancies=[]

        for filename,ecosystem in dependency_files.items():
            filepath = os.path.join(directory,filename)
            if os.path.exists(filepath):
                with open(filepath,"r") as f:
                    content=f.read()
                    found_dependancies.append(f"---{ecosystem}Dependancies({filename})--- \n{content}\n")
                    if not found_dependancies:
                        return "No common dependency files (e.g., requirements.txt, package.json) found"
                    return "\n".join(found_dependancies)
                
    
    def create_repo_analysis_agents(llm,retriever):
        """
        Creates and returns the specialized agents for the repository analysis
        """
        retrieval_tool=BaseTool(
            name="Repository Code Search",
            description="Searches the repository's codebase to fine relevant code snippets based on a query.",
            func=lambda q:retriever.invoke(q))

        dependency_tool=DependencyAnalysisTool()

        code_summarizer=Agent(
            role="Principal Code Summarizer",
            goal="Generate a high-level summary of the repository's purpose, architecture, and key components,",
            backstory=("""
                You are an expert software architect with years of experience in analyzing large codebases.
                Your task is to provide a clear and concise overview that helps developers quickly underand the project."""),
            verbose=True,
            llm=llm,
            tools=[retrieval_tool],
            allow_delegation=False)
        
        dependency_analyst=Agent(
            role="Software Dependency Analyst",
            goal="Identify and list all external dependencies of the project",
            backstory=("""
                You are a meticulous software engineer specializing in supply chain security.
                Your job is to find and analyze all the dependency files to create a comprehensive list of third-party libraries."""),
                verbose=True,
                llm=llm,
                tools=[dependency_tool],
                allow_delegation=False)
        
        security_auditor=Agent(
            role="Code Security Auditor",
            goal="Perform a basic security scan of the codebase to identify common vulnerabilities.",
            backstory=("""
                You are a sequrity expert with a keen eye for vulnerabilities.
                You scan code for potential issues like harcorded secrets and SQL injection patterns, providing a preliminary security assessment."""),
                verbose=True,
                llm=llm,
                tools=[dependency_tool],
                allow_delegation=False)
        
        documentation_generator=Agent(
            role="Technical Writer",
            goal="Synthesize the analysis from other agents into a comprehensive and well-structured final report",
            backstory=("""
                You are a professional techinal writer known for you ability to distill complex technical information
                into a clear, easy-to-understand documentation. You will take findings from your team and assemble them into a final report."""),
                verbose=True,
                llm=llm,
                allow_delegation=False)
        
        return {
            "summarizer":code_summarizer,
            "dependency_analyst":dependency_analyst,
            "security_auditor":security_auditor,
            "documentation_generator":documentation_generator
        }
