from langchain_community.llms import Cohere
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts  import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

def get_answer(question , text = open('sample.txt').read()) : 
    '''
    Function to get the answer to the question from the given text

    Args:
    question : str : The question to be asked
    text : str : The text from which the answer is to be extracted

    Returns:
    str : The answer to the question
    '''

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000 , 
        chunk_overlap = 200 , 
        length_function = len , 
        separators = [
            '\n' , '\n\n' , 
            '' , ' ' 

        ]
    )

    chunks = text_splitter.split_text(text = text)
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')

    vectorstore = FAISS.from_texts(chunks, embedding = embeddings)

    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    prompt_template = """Answer the question as precise as possible using the provided context. If the answer is
                    not contained in the context, say "answer not available in context" \n\n
                    Context: \n {context}?\n
                    Question: \n {question} \n
                    Answer:"""

    prompt = PromptTemplate.from_template(template=prompt_template)

    def format_docs(docs):
        '''
        Function to format the documents

        Args:
        docs : list : The list of documents

        Returns:
        str : The formatted documents
        '''
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(question):
        '''
        Function to generate the answer to the question

        Args:
        question : str : The question to be asked

        Returns:
        str : The answer to the question
        '''
        cohere_llm = Cohere(model="command", temperature=0.1, cohere_api_key = 'FELFXgLGfcqsy4eh4Q75dXNT7VyIQjKZmhkiIug3')
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | cohere_llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)

    return generate_answer(question)
