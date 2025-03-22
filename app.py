from orichain.llm import LLM, AsyncLLM
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

genai = AsyncLLM(
    model_name="us.amazon.nova-micro-v1:0",
    aws_access_key=os.getenv("AWS_BEDROCK_ACCESS_KEY"),
    aws_secret_key=os.getenv("AWS_BEDROCK_SECRET_KEY"),
    aws_region=os.getenv("AWS_BEDROCK_REGION"),
)

generator = genai.stream(user_message="hey", do_sse=False)


async def gg():
    async for i in generator:
        print(i)


asyncio.run(gg())
