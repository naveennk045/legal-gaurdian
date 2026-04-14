import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# print(os.environ.get("GROQ_API_KEY"))
llm = ChatGroq(
    model = "llama-3.3-70b-versatile",
    api_key = os.environ.get("GROQ_API_KEY")
)

messages = [
    (
        "system",
        "You are a helpful assistant and fun full friend make the user to getout from the  stress by cracking jokes",
    ),
    ("human", "I am failed in Exam"),
]
ai_msg = llm.invoke(messages)
print(ai_msg.content)

