from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


with open("API_KEY.txt", "r") as f:
    GOOGLE_API_KEY = f.read().strip()

print("Got API Key")

pdf_loader = PyPDFLoader("sample_docs/attention_is_all_you_need.pdf")
pages = pdf_loader.load_and_split()

print("Loaded PDF")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
# context = "\n\n".join(str(p.page_content) for p in pages)
# texts = text_splitter.split_text(context)
# 
# print("Split Text")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",google_api_key=GOOGLE_API_KEY)

print("Loaded Embeddings\n\n\n\n")

print(len(pages))

# texts = ['Once, a curious fox found a shimmering key in the forest. It led to a hidden door under an ancient oak. Inside, the fox discovered a magical library filled with glowing books. Each book whispered secrets of forgotten worlds. The fox read until dawn, learning about adventures and wisdom. When morning came, the fox left the library with a heart full of wonder and a mind brimming with knowledge. The key vanished, but the fox returned often, guided by the stories that now lived within. The forest, once ordinary, became a realm of endless possibilities.']

# vectorstore = Chroma("langchain_store", embeddings).add_texts(texts)
# vector_index = vectorstore.as_retriever()

vector_index = Chroma.from_documents(
    documents=pages,
    embedding=embeddings,
    persist_directory="data",
    collection_name="attention_is_all_you_need",
)
vector_index = vector_index.as_retriever()

print(vector_index)

model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY)

print("loaded Model")

template = """
Use the information provided in the "Context" field to answer the "Question".
If the context contains the necessary information to answer the question, provide a clear and concise answer.
If the context does not contain the information needed to answer the question, respond with: "I do not know the answer based on the provided context."
Do not repeat the information from the "Context" field in your answer.
context: {context}
question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)



qa_chain = RetrievalQA.from_chain_type(
    model,
    retriever=vector_index,
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

print("Made QA Chain")

question = "Describe the Multi-head attention layer in detail?"
question = "Who is the primeminister of India?"
result = qa_chain({"query": question})

print(f"\n\n\n {result}")