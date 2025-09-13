import os
import stat
from git import Repo
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader,DirectoryLoader
import shutil

#DictionaryMap {"FileExtentions":"LanguageName"}
LANGUAGE_MAP={
    ".py":"python",
    ".js":"js",
    ".java":"java",
    ".ts":"ts",
    ".html":"html",
    ".css":"css",
    ".md":"markdown",
    ".json":"json"
}

def clone_github_repo(repo_url:str , local_path:str="temp_rep")->str:
    """
    This fuction clones a public GitHub Repository to the given local directory.
    Deletes the directory if it already exists to ensure a frest start.
    """
    if os.path.exists(local_path):
        print(f"Directory {local_path} already exists. Removing it!")
        for root,dirs,files in os.walk(local_path):
            for name in files:
                filename=os.path.join(root,name)
                os.chmod(filename,stat.S_IRUSR | stat.S_IWUSR)
            for name in dirs:
                dirname=os.path.join(root,name)
                os.chmod(dirname,stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)
        
        shutil.rmtree(local_path)
    
    print(f"Cloning repository from{repo_url} to {local_path}....")
    Repo.clone_from(repo_url,local_path)
    print("Repository cloned Successfully.")
    return local_path

def load_and_chunk(repo_path:str)-> list:
    """Load the relevant files from the repository and splits them into semantic chunks.
    """
    print("Loading and chunking repository files...")
    documents=[]

    # Use DirectoryLoader to load all the files
    loader=DirectoryLoader(repo_path,glob="**/*",show_progress=True,use_multithreading=True, silent_errors=True)
    loaded_docs=loader.load()

    # Filter out files that we dont want to process
    filtered_docs=[doc for doc in loaded_docs 
                 if Path(doc.metadata.get("source","")).suffix in LANGUAGE_MAP
                 and ".git/" not in doc.metadata.get("source","")
                 ]
    print(f"Loaded {len(loaded_docs)} files, then filtered down to {len(filtered_docs)} relevant files.")

    all_splits=[]
    for doc in filtered_docs:
        file_extension=Path(doc.metadata["source"]).suffix
        language=LANGUAGE_MAP.get(file_extension)

        if language:
            splitter=RecursiveCharacterTextSplitter.from_language(
                language=language,chunk_size=1000,chunk_overlap=100
            )
            splits=splitter.split_documents([doc])
            all_splits.extend(splits)
    
    print(f"Split documents into {len(all_splits)} chunks")
    return all_splits
