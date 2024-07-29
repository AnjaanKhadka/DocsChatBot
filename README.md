# DocsChatBot

A chatbot built using LangChain and Gemini that answers user queries from documents and includes a conversational form for collecting user information. Features date extraction and validation for user inputs.

## Functionalities

### Document QA part

It is a implementation of document retrival type chatbot that can answer based on information available in a python file. Currently I am using [Attention is all you need](https://arxiv.org/pdf/1706.03762) paper. This file is kept in [sample_docs](https://github.com/AnjaanKhadka/DocsChatBot/blob/main/sample_docs/) folder as [attention_is_all_you_need.pdf](https://github.com/AnjaanKhadka/DocsChatBot/blob/main/sample_docs/attention_is_all_you_need.pdf).

You can Change the content of file or use multiple files with simple tweek in the code.

### Booking part (Conversation Form)

Now Booking ststem is also implemented. Booking session is initiated by simply asking to make a booking in Natural language.

Booking system can extract information from the natural conversation between user and ChatBot. This System can efficiently collect information in conversational Form.

## Installation

Clone the repository in a folder.

    git clone https://github.com/AnjaanKhadka/DocsChatBot.git

Get in the direcory

    cd DocsChatBot

All dependencies are kept in [requirements.txt](https://github.com/AnjaanKhadka/DocsChatBot/blob/main/requirements.txt) file.

    pip install -r requirements.txt

Beside requirements, you need to have Gemini API key. You can create a free [Gemini API key](https://aistudio.google.com/app/apikey) for testing purpose.

After getting the key, Copy your API key and paste it in your API_KEY.txt file in the primary direcory.

    echo Your_API_KEY > API_KEY.txt

## Execution

I am using Streamlit to create interactive chatbot. Run the [main.py](https://github.com/AnjaanKhadka/DocsChatBot/blob/main/main.py) with streamlit.

    streamlit run main.py

## Demo Run

### Document QA System

https://github.com/user-attachments/assets/33e52b83-2993-48ea-a998-423d17929238

### Triggering the Booking system



### Collecting information


