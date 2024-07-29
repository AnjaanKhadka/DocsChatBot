
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
# from langchain_chroma import Chroma
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import re
from langchain_core.output_parsers import JsonOutputParser
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

safety_settings = {
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
}



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
    model = ChatGoogleGenerativeAI(model="gemini-pro",google_api_key=GOOGLE_API_KEY,safety_settings=safety_settings)

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

User can also ask you to place a booking or make a reservation or ask about the booking, In such cases, reply to the message with "START_BOOKING_PROCESS" and nothing more.

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


class Details(BaseModel):
    first_name: str = Field(description="This is the first name of user",title="first_name")
    last_name: str = Field(description="This is the last name of user",title="last_name")
    phone: str = Field(description="This is the Phone Number or Contact Number of the user",title="phone")
    email: str = Field(description="This is the Email Address of the user",title="email")
    yyyy: int = Field(description="This is the Year of Booking date in YYYY format",title="yyyy")
    mm: int = Field(description="This is the Month of Booking date in MM format",title="mm")
    dd: int = Field(description="This is the Day of Booking date in DD format",title="dd")

    

class PersonalDetails:
    def __init__(self, first_name='', last_name='', phone='', email='', yyyy='', mm='', dd=''):
        self.first_name = first_name
        self.last_name = last_name
        self.phone = phone
        self.email = email
        self.yyyy = yyyy
        self.mm = mm
        self.dd = dd
        self.is_complete = False

    def fill(self, new_data):
        fields = ['first_name', 'last_name', 'phone', 'email', 'yyyy', 'mm', 'dd']
        # new_data = new_data['args']
        for key, value in new_data.items():
            if key in fields:
                if value:
                    setattr(self, key, value)
        # Define a regular expression pattern for each field
        # patterns = {
        #     'first_name': r'first_name:\s*(.*)',
        #     'last_name': r'last_name:\s*(.*)',
        #     'phone': r'phone:\s*(.*)',
        #     'email': r'email:\s*(.*)',
        #     'yyyy': r'yyyy:\s*(\d{4})',
        #     'mm': r'mm:\s*(\d{2})',
        #     'dd': r'dd:\s*(\d{2})'
        # }

        # # Extract information using the defined patterns
        # for key, pattern in patterns.items():
        #     # print(type(content))
        #     match = re.search(pattern, content)
        #     if match:
        #         setattr(self, key, match.group(1))
        
        # pattern = re.compile(r'first_name:\s*(\S*)\nlast_name:\s*(\S*)\nphone:\s*(\S*)\nemail:\s*(\S*)\nyyyy:\s*(\d{4})\nmm:\s*(\d{2})\ndd:\s*(\d{2})')
        # match = pattern.search(content)
        
        
        
        # if match:
        #     self.first_name = self.first_name or match.group(1)
        #     self.last_name = self.last_name or match.group(2)
        #     self.phone = self.phone or match.group(3)
        #     self.email = self.email or match.group(4)
        #     self.yyyy = self.yyyy or match.group(5)
        #     self.mm = self.mm or match.group(6)
        #     self.dd = self.dd or match.group(7)
        
        # # Check if all fields are filled
        self.is_complete = all(getattr(self, key) for key in ['first_name', 'last_name', 'phone', 'email', 'yyyy', 'mm', 'dd'])
         

    def date_info(self):
        return f"{self.yyyy}/{self.mm}/{self.dd}"

    def known_fields(self):
        return '\n'.join(f'{key}: {getattr(self, key)}' for key in ['first_name', 'last_name', 'phone', 'email', 'yyyy', 'mm', 'dd'] if getattr(self, key))
    
    def unknown_fields(self):
        return '\n'.join(key for key in ['first_name', 'last_name', 'phone', 'email', 'yyyy', 'mm', 'dd'] if not getattr(self, key))
        
        
    def conversational_Query(self,prompt):
        template = '''
            You are an Information Collecting AI. You need to collect personal information from the user.
            You are supposed to understand the user's input and extract the personal information from it.
            Information to be collected:
            - First Name
            - Last Name
            - Phone Number
            - Email Address
            - Date of Booking (YYYY-MM-DD)
            
            User can provide dates in relative format, make sure to convert it to the correct format.
            For reference, today is: 
            {}
            
            User Input: {}
            
            Response format: JSON with the following fields:
            first_name: ""
            last_name: ""
            phone: ""
            email: ""
            yyyy: ""
            mm: ""
            dd: ""

        
        '''
        template = template.format(
            datetime.now().strftime("%Y-%m-%d %A"),
            prompt
        )
        
        parser = JsonOutputParser(pydantic_object=Details)
        info = model.invoke(template)
        
        info = parser.parse(info.content)
        
        
        
        print(info)
        
        self.fill(info)
        
        if self.is_complete:
            return f"Thank You {self.first_name}, Booking is confirmed for {self.date_info()}"
        
        template = '''
You are supposed to gather information for a booking for a dummy project. (Name, Contact and Date of Booking)

{}

Required Information:
    The information still needed from the user is:
        {}

Dates for the booking are also acceptable in relative format, but you do not have to specify that in the prompt unless asked.

Instructions for the model:

    - Do not ask for information that is not needed.
    - Its important to ask the questions in a natural way like a human would.
    - Do not specify the collected information in the prompt.
    - If information are partially collected, ask for the remaining information one by one in polite manner.
    - Dont ask the date with year, month and day separately. Ask for the date in a single question, when asking for the date.
    - If all the information is collected, just thank the user for placing the booking.
    - Ask for only one information at a time among Name, Contact or booking date.

Users are allowed to enter multiple information in a single sentence.
        
    
    
One Question to ask:
    '''
        
#         template = '''
# You are an AI Assistant model tasked for asking for information for making a booking from the user in natural way.
# So far Information collected:
# {}

# You need to collect the following information from the user:
# {}

# Do not ask for the information that is already provided by the user or is not needed.

# Make sure to address the user with their name if and only if available.

# Ask one question at a time and wait for the user's response before asking the next question.

# Your prompt should be in the form of a question.
            
#         '''

        name = ""
        if self.first_name:
            name = f"You can address user with the name {self.first_name}"
        
        
        
        template = template.format(
            name,
            self.unknown_fields(),
            # prompt,
            # datetime.now().strftime("%Y-%m-%d %A")
        )
        
        print(f"\n\ntemplate:\n\n {template}\n\n")
        # safe = {
        # "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        # "threshold": "BLOCK_NONE",
        # }
        response = model.invoke(template,safety_settings=safety_settings)
        
        print(f"You are {self.first_name} {self.last_name}, you can be contacted at {self.phone} or {self.email}. Your booking is confirmed for {self.date_info()}")
        
        print(f"\n response: {response}")
        
        return response.content
        






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
