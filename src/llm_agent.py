import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4", temperature=0.2)

def process_task(task: str, df: pd.DataFrame = None):
    if df is not None:
        agent = create_pandas_dataframe_agent(llm, df, verbose=True)
        try:
            response = agent.run(task)
            return {"result": response}
        except Exception as e:
            return {"error": str(e)}
    else:
        return {"result": llm.predict(task)}
