import os

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Missing OPENAI_API_KEY. Add it to your environment or a .env file before running this demo."
    )

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
llm = ChatOpenAI(model=model_name, temperature=0)

result = llm.invoke("What is the capital of India?")

print(result.content)
