from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
)
#now we need to build prompt template it will have 3 variables attack type, confidence, and top features 
template = PromptTemplate(
    input_variables=["attack_type", "confidence", "top_features"],
    template="""
        You are a senior cybersecurity analyst.
    
        A network intrusion detection system has detected the following:
        - Attack Type: {attack_type}
        - Confidence Score: {confidence}%
        - Key Network Features that triggered this: {top_features}
    
        Please provide:
        1. A clear explanation of what this attack is
        2. Why these specific features indicate this attack
        3. The potential damage this attack could cause
        4. Immediate steps the network administrator should take
    
        Keep the explanation clear enough for a junior engineer to understand.
    """,
)
chain = template | llm
def generate_explanation(attack_type, confidence, top_features):
    response = chain.invoke({
        "attack_type": attack_type,
        "confidence": confidence,
        "top_features": top_features
    })
    return response.content


def ask_security_chatbot(question):
    #building the chat bot first load the chroma vector we stored in building_vectorstore.py
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory='chroma_db', embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    response = llm.invoke( f"You are a cybersecurity expert. Answer this question using the context below.\n\nContext: {context}\n\nQuestion: {question}" )
    return response.content

# print(ask_security_chatbot("What does OWASP say about broken access control?"))