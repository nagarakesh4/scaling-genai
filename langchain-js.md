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


### Programming
```
import { ChatOpenAI } from "@langchain/openai";
import { HumanMessage } from "@langchain/core/messages";

const model = new ChatOpenAI({
    modelName: "gpt-3.5-turbo-1106"
});

await model.invoke([
    new HumanMessage("Tell me a fact")
]);
```

#### Building block: Prompt Template 
- calling models in isolation are useful as above
- convenient to factor logic into model inputs into reasonable parametrized components
- responsible for formatting user inputs that our model calls
```
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromTemplate(
    `What are three good names for a company that makes {product}?`
)

await prompt.format({
    product: "colorful socks"
});  //string output - "Human: What are three good names for a company that makes colorful socks?" - this can be passed directly to LLM (not chat models)

//pass to Chatmodels
await prompt.formatMessages({
    product: "colorful socks"
}); //this will convert to HumanMessage{...content: ... } format for chat models, 

//to create both system and human messages for LLMs
import { 
  SystemMessagePromptTemplate, 
  HumanMessagePromptTemplate 
} from "@langchain/core/prompts";

const promptFromMessages = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate(
    "You are an expert at picking company names."
  ),
  HumanMessagePromptTemplate.fromTemplate(
    "What are three good names for a company that makes {product}?"
  )
]);

await promptFromMessages.formatMessages({
    product: "shiny objects"
});

//using tuples with role in template: the above can be converted to, which also generates array of system and human messages
const promptFromMessages = ChatPromptTemplate.fromMessages([
  ["system", "You are an expert at picking company names."],
  ["human", "What are three good names for a company that makes {product}?"]
]);

await promptFromMessages.formatMessages({
  product: "shiny objects"
});

//the above can be passed to history, to inject history directly to prompt
```

#### LCEL
- elegant way of using prompts and models (instead of sending like above)
- composable syntax for chaining langchain models together
- objects that are compatible with LCEL are runnables
- can create simple chain

```
const chain = prompt.pipe(model);
await chain.invoke({
    product: "colorful socks"
}); //suggestes colorful socks names in chat messages
```

#### Output parser
```
import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

//piping the above output parser (string) to model's prompt
const nameGenerationChain = prompt.pipe(model).pipe(outputParser);

await nameGenerationChain.invoke({
    product: "fancy cookies"
}); // prints only the string "1. Gourmet Cookie Co.\n2. DelicateDesserts\n3. Exquisite Eats Bakery"
```

- visual implementation of above [link](https://smith.langchain.com/public/9fd7689f-23c6-4ea1-8a76-fea1b5f8d5db/r/19a04da8-832f-4965-9587-72836a80dd4c)

#### Runnable sequence
- For more complex pipe chains, create an array of runnables

```
import { RunnableSequence } from "@langchain/core/runnables";

const nameGenerationChain = RunnableSequence.from([
    prompt,
    model,
    outputParser
]) ; //same as prev ex, but in a scalable array format

await nameGenerationChain.invoke({
    product: "fancy cookies"
}); //same response as above in string parser
```

#### Methods from LCEL 
##### Streaming (for Frontend)
- Another method from LCEL
- All sequences of runnables get a `.stream()` in an iterable stream
- rather than waiting for entire response, can show chunks as they get processed 
```
//using the above nameGenerationChain that has array of runnables implemented will help to get the string output instead of models response
//use .stream()
const stream = await nameGenerationChain.stream({
  product: "really cool robots",
}); //get indiv chunks
 
//loop over returned stream chunks
for await (const chunk of stream) {
    console.log(chunk);
}
```
#### Batch
- multiple/concurrent operations simultaneously
```
const inputs = [
    { product: "large calculators" },
    { product: "alpaca wool sweaters" }
];

//use .batch()
await nameGenerationChain.batch(inputs);

//batches output
[
  "1. MegaMath Calculators\n2. SummitTech Calculators\n3. CalculaGiant Company",
  "1. Alpaca Luxe\n2. Andean Elegance\n3. Woolly Alpaca Co."
]
```

### RAG
- Document loaders: load documents from source (PDF, db, web)
- Splitting: split docs into chunks to fit into LLM context window to avoid distraction (while predicting)
- Storage: embed chunks into vector store to allow for later retrieval based on input queries
- Query: Now when user wants to access the data (performs a query) we retrieve the relevant previously split chunks and send the final output as context


#### Langchain: Document loaders
- <img width="200" height="200" alt="image" src="https://github.com/nagarakesh4/genai-scale-notes/assets/4505218/918f49f9-df0a-4d7f-9c1b-0ed3796f3f78">

##### GitHub Reader
```
import "dotenv/config";
import { GithubRepoLoader } from "langchain/document_loaders/web/github";
// Peer dependency, used to support .gitignore syntax
import ignore from "ignore";

// Will not include anything under "ignorePaths"
// Will not include anything under "ignorePaths"
const loader = new GithubRepoLoader(
  "https://github.com/nagarakesh4/web",
  { branch: 'master', recursive: false, ignorePaths: ["*.md", "yarn.lock"] }
);

//load the docs
const docs = await loader.load();
console.log(docs.slice(0, 3));
//prints the file name, content (metadata) of the above repo in LLM response format
[
  Document {
    pageContent: "<!DOCTYPE html>\n" +
      '<html lang="en">\n' +
      "<head>\n" +
      '    <meta charset="UTF-8">\n' +
      '    <meta name="viewport" content'... 477 more characters,
    metadata: {
      source: "index.html",
      repository: "https://github.com/nagarakesh4/web",
      branch: "master"
    }
  },
  Document {
    pageContent: "//comments helps others to understand your code better\n" +
      "//single line comments are denoted with doubl"... 1867 more characters,
    metadata: {
      source: "index.js",
      repository: "https://github.com/nagarakesh4/web",
      branch: "master"
    }
  }
]
```

##### PDF Reader
```
// Peer dependency
import * as parse from "pdf-parse";
import { PDFLoader } from "langchain/document_loaders/fs/pdf";

const loader = new PDFLoader("./data/MachineLearning-Lecture01.pdf");

const rawCS229Docs = await loader.load();

# log first 5 pages
console.log(rawCS229Docs.slice(0, 5));
//similar to above shows metadata / content of the pdf
```


#### Langchain: Splitting
- semantically related ideas in the same chunk so that LLM gets the entire self-contained idea without any distraction
- split depends on what we are loading, e.g., js documents are split by fun impl so that LLM is aware of the (self-contained) context
- RecursiveCharacterTextSplitter splits on paragraphs is a good start

- Using JS example
```
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";


const splitter = RecursiveCharacterTextSplitter.fromLanguage("js", { //understands that common 'js' features are the chunks 
  chunkSize: 32, //very small
  chunkOverlap: 0, //useful to serve chunks across each other
});

const code = `function helloWorld() {
console.log("Hello, World!");
}
// Call the function
helloWorld();`;

await splitter.splitText(code);
//naturally split (as per js coding)
[
  "function helloWorld() {",
  'console.log("Hello, World!");\n}',
  "// Call the function",
  "helloWorld();"
]

//if we split naively (like spaces), we get chunks split differently which makes LLM job difficult on generation
import { CharacterTextSplitter } from "langchain/text_splitter";

const splitter = new CharacterTextSplitter({
  chunkSize: 32,
  chunkOverlap: 0,
  separator: " " //space as naive character splitter
});

await splitter.splitText(code);
//LLM will have little hard time to understand below
[
  "function helloWorld()",
  '{\nconsole.log("Hello,',
  'World!");\n}\n// Call the',
  "function\nhelloWorld();"
]

//increasing chunk size and chunk overlap will set full context for LLM

```

- Using on PDF
```
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 512,
  chunkOverlap: 64,
});

const splitDocs = await splitter.splitDocuments(rawCS229Docs);

console.log(splitDocs.slice(0, 5));
```

#### Langchain: Embed in Vector store 
- we embed chunks in Vector store for easier search and query based on input query
- text embedding model: OpenAIEmbedding
- Vector store: specialized type of database with natural language search capabilities
- when user comes with query, search vector store for an embedding similar to the query asked and fetch relevant chunks
