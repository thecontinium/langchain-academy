# Using Ollama 

The code for studio and notebooks has been updated to use Ollama with llama3.1-tools rather than OpenAI.

Follow all the general instructions for setup and then:

## llama3.1-tools

llama3.1 has tooling but does not default to a normal answer if the tooling is not appropriate.
To acheive a similar response to OpenAI we create llama3.1-tools

1. Install Ollama
2. `Ollama pull llama3.1` ( which has a tools interface )
3. Creating llama3.1-tools

To use with Ollama and make llama3.1 answer normally when a tool is not appropriate I followed the advice [here](https://github.com/ollama/ollama/issues/6127#issuecomment-2264291170) and [here](https://github.com/ollama/ollama/issues/6127#issuecomment-2379762636) and added the following to the 'modelfile' as suggested:

    Analyse the given prompt and decided whether or not it can be answered by a tool.  If it can, use the following functions to respond with a JSON for a function call with its proper arguments that best answers the given prompt.  Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.
    NEVER make up your own parameter values as tool function arguments, like 'city=London'!
    NEVER use tool functions if not asked, instead revert to normal chat!

See [here](./Modelfile)

## Using Ollama With Lang Studio

Use [this](https://github.com/langchain-ai/langgraph-studio/issues/112) to get ollama working inside the docker image.
To get ollama working from within the Lang Studio which uses Docker we need to
explicitly define the base url mapped to the external resource. The code has been updated other use:

``` python
llm = ChatOllama( model="llama3.1-tool", temperature=0,base_url="http://host.docker.internal:11434") # other params...)
```
