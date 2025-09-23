import streamlit as st
from repository_processor import clone_github_repo, load_and_chunk
from vector_store_manager import VectorStoreManager
from main_crew import RepoAnalysisCrew

st.set_page_config(page_title="Repo-Intellect",layout="wide")
st.title("🧠 Repo-Intellect: AI GitHub Repository Analysis Agent")

def _strip_md_fences(text: str) -> str:
    """Remove surrounding ``` or ```markdown fences if present."""
    if not text:
        return text
    s = text.strip()
    if s.startswith("```"):
        # drop the first fence line (``` or ```markdown)
        nl = s.find("\n")
        if nl != -1:
            s = s[nl+1:]
        # drop trailing fence
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()

st.sidebar.header("Instructions")
st.sidebar.info("""
            1.Enter a public GitHub repository URL.\n
            2.Provide a high-level query or question about the repository.\n
            3.Click 'Analyze Repository' to start the analysis. \n
            \n
            This process may take several minutes depending on the repositoy size.

                """)
repo_url=st.text_input("Enter a GitHub Repository URL:", placeholder="https://github.com/user/repo")
user_query=st.text_area("What do you want to know abouth the shared repository?", placeholder="e.g., Provide an analysis of the repository.")

if st.button("Analyze Repository"):
    if not repo_url:
        st.error("Please provide a repository URL")
    elif not user_query:
        st.error("Please provide a query")
    else:
        try:
            with st.spinner("Step 1/4: Cloning the shared repository."):
                repo_path=clone_github_repo(repo_url=repo_url)
                st.success("Repository cloned successfully!")

            with st.spinner("Step 2/4: Processing and chunking files..."):
                documents=load_and_chunk(repo_path)
                st.success("Files processed and chunked!")
            
            with st.spinner("Step 3/4: Creating vector store and retriever..."):
                collection_name=f"repo_{repo_url.split('/')[-1]}"
                vector_store_manager=VectorStoreManager(collection_name=collection_name)
                vector_store_manager.populate_vector_store(documents)
                retreiver=vector_store_manager.get_retriever()
                st.success("Vector store created!")
            
            with st.spinner("Step 4/4: The AI crew is analyzing the repository. This may take a while."):
                analysis_crew = RepoAnalysisCrew(retriever=retreiver)
                result = analysis_crew.run(repo_path=repo_path, user_query=user_query)
                st.success("Analysis Complete!")

            # Convert CrewOutput to string if needed
            if isinstance(result, str):
                md = result
            else:
                md = (
                    getattr(result, "raw", None)
                    or getattr(result, "final_output", None)
                    or getattr(result, "output", None)
                    or str(result)
                )
            st.markdown(md)

        except Exception as e:
            st.error(f"An error occured: {e}")
            