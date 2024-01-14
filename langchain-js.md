# Langchain.js

### Intro
- A RAG application needs LLM to perform a task, retrieve the text, tune the prompt, pass the LLM text output to structured for downstream steps of app
- Tools that connect the above workflow and also tune the workflow for e.g., changing the LLM models, tie multiple chains, these tools are called as Orchestrators.
- Langchain are popular ochestrator for LLM applications and helps to build them quickly

### Elements
- Data loaders - pull data from sources such as PDFs etc that augment the LLM generation
- Parsers - LLM operate with Natural language, while programming operate with formatted data, parsers extract and format Natural language output to create structured forms
- Prompts - Used to provide context for LLM
- Models - Provide abstraction layer on LLM (to write apps that are not vendor specific)
- Modules to support RAG - text splitters / integrations w/ vector stores
- LCEL - LangChain Expression lang - compose complex chain of these modules

### LCEL 
- composes chains of components
- components that implement this lang are called runnables
- allows input/output types, methods: invoke,stream, batch to use with LLM
- fallbacks/parallelism are available out-of-box
- 
