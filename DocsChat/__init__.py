from .DocsChat import initial_load

qa_chain = initial_load()

def query(question =None):
    if not question:
        print("Replying to generic question, What is the attention mechanism?")
        question = "What is the attention mechanism?"
    result = qa_chain({"query": question})
    return result

