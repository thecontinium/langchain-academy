Using Ollama with llama3.1-tools



To use with Ollama and make llama3.1 answer normally when a tool is not
appropriate I followed the advice in the following issue:

https://github.com/ollama/ollama/issues/6127#issuecomment-2264291170

https://github.com/ollama/ollama/issues/6127#issuecomment-2379762636


and added the following to the 'modelfile' as suggested:

Analyse the given prompt and decided whether or not it can be answered by a tool.  If it can, use the following functions to respond with a JSON for a function call with its proper arguments that best answers the given prompt.  Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.
NEVER make up your own parameter values as tool function arguments, like 'city=London'!
NEVER use tool functions if not asked, instead revert to normal chat!


Use https://github.com/langchain-ai/langgraph-studio/issues/112 to get ollama
working inside the docker image.

llm = ChatOllama( model="llama3.1-tool", temperature=0,base_url="http://host.docker.internal:11434") # other params...)
