
import os
from dotenv import load_dotenv
from groq import Groq
#Complete code for rag.py
from langchain.llms import LlamaCpp
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_groq import ChatGroq
from flask import Flask,render_template,request,jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index4.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    # Here you would integrate your chatbot logic
    

    


    bot_response = get_bot_response(user_input)
    return jsonify(response=bot_response)

def get_bot_response(user_input):
    # Placeholder for chatbot logic
    load_dotenv()

    client = ChatGroq(
        api_key=os.environ.get("GROQ_API_KEY"),
        model="llama3-70b-8192",
        temperature=0.75,
    
        max_tokens=2000,
    
        top_p=0.9,
    
        verbose=True,
    )

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma(
    persist_directory="data1",
    embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

    template = """

    You are experienced Salesworx Assistant with deep knowledge on the product and its usage . Provide direct, concise answers to questions. Do not use phrases such as "Based on the provided context," "According to the document," or similar references. Simply answer the questions directly .If a question asks for information outside the provided context, respond with "I'm sorry, I don't have information on that topic. Please respond in a friendly manner.provide appropriate emoji if needed.
    
    

    {user_manual}
    

    Questions:{query}

    Answer:


    """

    prompt = ChatPromptTemplate.from_template(template=template)

    chain = (
        {"user_manual": retriever, "query": RunnablePassthrough()}
        | prompt
        | client
        | StrOutputParser()

    )
    response_chunks = []
    for chunk in chain.stream(user_input):
        response_chunks.append(chunk)
    
    return ''.join(response_chunks)

if __name__ == '__main__':
    app.run(debug=True)
