
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


qa_chain = None


def get_api():
    global GOOGLE_API_KEY
    with open("API_KEY.txt", "r") as f:
        GOOGLE_API_KEY = f.read().strip()

def get_embeding():
    global embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)
    
def load_docs():
    pdf_loader = PyPDFLoader("sample_docs/attention_is_all_you_need.pdf")
    pages = pdf_loader.load_and_split() 
    return pages

def load_model():
    global model
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY)

def create_vector_index(docs):
    # print(docs)
    global vector_index
    vector_index = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./data",
        collection_name="attention_is_all_you_need",
    )
    # vector_index.persist()
    vector_index = vector_index.as_retriever()



def create_qa_chain():
    global qa_chain
    template = """
Context:

The following is the context information the model has access to:
{context}

Question:

User's query: {question}

Instructions for the model:

If the context contains the information needed to answer the question, provide a clear and concise response based on the context without repeating the context verbatim.
Engage in basic conversational interactions such as greetings and farewells naturally and appropriately.
    """
    # template = """ 
    # Context:
    # {context}

    # Question:
    # {question}

    # Instructions for the Model:

    # Use the information provided in the "Context" field to answer the "Question".
    # If the context contains the necessary information to answer the question, provide a clear and concise answer.
    # If the context does not contain the information needed to answer the question, respond with: "I do not know the answer based on the provided context."
    # Do not repeat the information from the "Context" field in your answer.
    
    # Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    qa_chain = RetrievalQA.from_chain_type(
        model,
        retriever=vector_index,
        return_source_documents=False,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )


def initial_load():
    get_api()
    # print("API Key Loaded")
    docs = load_docs()
    # print("Docs Loaded")
    get_embeding()
    # print("Embedding Loaded")
    create_vector_index(docs)
    # print("Vector Index Created")
    load_model()
    # print("Model Loaded")
    create_qa_chain()
    # print("QA Chain Created")
    return qa_chain
