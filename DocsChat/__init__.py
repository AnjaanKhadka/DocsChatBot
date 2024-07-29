from .DocsChat import initial_load
from .DocsChat import PersonalDetails
# from .DocsChat import Conversational_Query

qa_chain = initial_load()

is_booking = False

personal_info = PersonalDetails()

def query(question =None):
    global is_booking
    print(question)
    
    if personal_info.is_complete:
        is_booking = False
    
    if not is_booking:
        if not question:
            print("Replying to generic question, What is the attention mechanism?")
            question = "What is the attention mechanism?"
        result = qa_chain({"query": question})['result']
        if result == "START_BOOKING_PROCESS":
            is_booking = True
        else:    
            return result
    
    # if personal_info.is_complete:
    #     is_booking = False
    #     return f"Thank you {personal_info.first_name}, your booking is confirmed for {personal_info.date_info}"    
    # else:
    if not personal_info.is_complete:
        return personal_info.conversational_Query(question)



