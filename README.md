# llama-chat_w_pdf
Chat with your PDF using LLama3

## Create a Simple RAG on your local

Create a simple RAG system on your local system using LangChain, Ollama and Streamlit

# Pre-Requisites

Docker - Learn how to install docker 
- [MacOS](https://docs.docker.com/desktop/install/mac-install/)
- [Windows](https://docs.docker.com/desktop/install/windows-install/)
- [Ubuntu](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-docker-on-ubuntu-20-04)

# Installation
 - Install Ollama on your local to use this. This can be done using docker. You can read the article on my [blog](https://testerops.com/2024/05/02/run-ollama-models-on-local/) where I have shown how you can run Ollama models on local
 - Clone the repository - `git clone <repo_link>`
 - Install the requirements by using `pip install -r requirements.txt` [Note please use `Python <3.13` if using PyCharm as langchain installation is failing on PyCharm due to a bug]
 - To run the code - `streamlit run main.py`

# Use
Upload a PDF ( should be `< 200 MB` in size) and then you can ask questions around the contents of the PDF

# Packages used
- langchain
- langchain_community
- streamlit
- streamlit_chat
- chromadb
- pypdf
- fastembed


