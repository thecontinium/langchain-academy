# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: langchain-academy
#     language: python
#     name: langchain-academy
# ---

# %% [markdown]
# [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-0/basics.ipynb) [![Open in LangChain Academy](https://cdn.prod.website-files.com/65b8cd72835ceeacd4449a53/66e9eba12c7b7688aa3dbb5e_LCA-badge-green.svg)](https://academy.langchain.com/courses/take/intro-to-langgraph/lessons/56295530-getting-set-up-video-guide)

# %% [markdown]
# # LangChain Academy
#
# Welcome to LangChain Academy! 
#
# ## Context
#
# At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There’s a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 
#
# In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we’ve worked with our users to put agents into production, we’ve learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state. 
#
# To tackle this problem, we’ve built [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph’s core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.
#
# ## Course Structure
#
# The course is structured as a set of modules, with each module focused on a particular theme related to LangGraph. You will see a folder for each module, which contains a series of notebooks. A video will accompany each notebook to help walk through the concepts, but the notebooks are also stand-alone, meaning that they contain explanations and can be viewed independently of the videos. Each module folder also contains a `studio` folder, which contains a set of graphs that can be loaded into [LangGraph Studio](https://github.com/langchain-ai/langgraph-studio), our IDE for building LangGraph applications.
#
# ## Setup
#
# Before you begin, please follow the instructions in the `README` to create an environment and install dependencies.
#
# ## Chat models
#
# In this course, we'll be using [Chat Models](https://python.langchain.com/v0.2/docs/concepts/#chat-models), which do a few things take a sequence of messages as inputs and return chat messages as outputs. LangChain does not host any Chat Models, rather we rely on third party integrations. [Here](https://python.langchain.com/v0.2/docs/integrations/chat/) is a list of 3rd party chat model integrations within LangChain! By default, the course will use [ChatOpenAI](https://python.langchain.com/v0.2/docs/integrations/chat/openai/) because it is both popular and performant. As noted, please ensure that you have an `OPENAI_API_KEY`.
#
# Let's check that your `OPENAI_API_KEY` is set and, if not, you will be asked to enter it.

# %%
# %%capture --no-stderr
# %pip install --quiet -U langchain_openai langchain_core langchain_community tavily-python

# %%
import os, getpass

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

#_set_env("OPENAI_API_KEY")


# %% [markdown]
# [Here](https://python.langchain.com/v0.2/docs/how_to/#chat-models) is a useful how-do for all the things that you can do with chat models, but we'll show a few highlights below. If you've run `pip install -r requirements.txt` as noted in the README, then you've installed the `langchain-openai` package. With this, we can instantiate our `ChatOpenAI` model object. If you are signing up for the API for the first time, you should receive [free credits](https://community.openai.com/t/understanding-api-limits-and-free-tier/498517) that can be applied to any of the models. You can see pricing for various models [here](https://openai.com/api/pricing/). The notebooks will default to `gpt-4o` because it's a good balance of quality, price, and speed [see more here](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4-gpt-4-turbo-gpt-4o-and-gpt-4o-mini), but you can also opt for the lower priced `gpt-3.5` series models. 
#
# There are [a few standard parameters](https://python.langchain.com/v0.2/docs/concepts/#chat-models) that we can set with chat models. Two of the most common are:
#
# * `model`: the name of the model
# * `temperature`: the sampling temperature
#
# `Temperature` controls the randomness or creativity of the model's output where low temperature (close to 0) is more deterministic and focused outputs. This is good for tasks requiring accuracy or factual responses. High temperature (close to 1) is good for creative tasks or generating varied responses. 

# %%
#from langchain_openai import ChatOpenAI
#gpt4o_chat = ChatOpenAI(model="gpt-4o", temperature=0)
#gpt35_chat = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
from langchain_ollama import ChatOllama
ollama_chat = ChatOllama(
    model="llama3.1",
    temperature=0,
    # other params...
)

# %% [markdown]
# Chat models in LangChain have a number of [default methods](https://python.langchain.com/v0.2/docs/concepts/#runnable-interface). For the most part, we'll be using:
#
# * `stream`: stream back chunks of the response
# * `invoke`: call the chain on an input
#
# And, as mentioned, chat models take [messages](https://python.langchain.com/v0.2/docs/concepts/#messages) as input. Messages have a role (that describes who is saying the message) and a content property. We'll be talking a lot more about this later, but here let's just show the basics.

# %%
from langchain_core.messages import HumanMessage

# Create a message
msg = HumanMessage(content="Hello world", name="Lance")

# Message list
messages = [msg]

# Invoke the model with a list of messages 
ollama_chat.invoke(messages)

# %% [markdown]
# We get an `AIMessage` response. Also, note that we can just invoke a chat model with a string. When a string is passed in as input, it is converted to a `HumanMessage` and then passed to the underlying model.
#

# %%
ollama_chat.invoke("hello world")

# %%
#gpt35_chat.invoke("hello world")

# %% [markdown]
# The interface is consistent across all chat models and models are typically initialized once at the start up each notebooks. 
#
# So, you can easily switch between models without changing the downstream code if you have strong preference for another provider.
#

# %% [markdown]
# ## Search Tools
#
# You'll also see [Tavily](https://tavily.com/) in the README, which is a search engine optimized for LLMs and RAG, aimed at efficient, quick, and persistent search results. As mentioned, it's easy to sign up and offers a generous free tier. Some lessons (in Module 4) will use Tavily by default but, of course, other search tools can be used if you want to modify the code for yourself.

# %%
_set_env("TAVILY_API_KEY")

# %%
from langchain_community.tools.tavily_search import TavilySearchResults
tavily_search = TavilySearchResults(max_results=3)
search_docs = tavily_search.invoke("What is LangGraph?")

# %%
search_docs

# %%
