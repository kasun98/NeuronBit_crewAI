from crewai import Agent
from tools import tool_serper, tool_investing, tool_binancenews, tool_cmc, tool_csv
from dotenv import load_dotenv
load_dotenv()
from langchain_google_genai import ChatGoogleGenerativeAI
import os


## call the gemini models
llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))

# Creating a TA agent with memory and verbose mode

technical_analysis_agent = Agent(
    role="Technical Analyst",
    goal='Analyze Bitcoin price movements using technical indicators and chart patterns within {L_date} and {Date}. Today is {Date}',
    memory=True,
    backstory=(
        """With a keen eye for market trends and patterns, you excel in
        analyzing price movements and technical indicators to provide
        insights on potential future price actions. Provide the 1day market will be bullish or Bearish based on data to day traders."""
    ),
    tools=[tool_cmc, tool_serper, tool_csv],
    llm=llm,
    allow_delegation=True,
    max_iter=10
)


# creating a FA agent 
fundamental_analysis_agent = Agent(
    role="Fundamental Analyst",
    goal='Analyze Bitcoin price using fundamental economic factors and last 24 hours news. Today is {Date}',
    memory=True,
    backstory=(
        """With a deep understanding of market forces and economic indicators,
        you evaluate the intrinsic value of assets, considering broader
        economic conditions and industry trends. Provide the 1day market will be bullish or Bearish based on data to day traders"""
    ),
    tools=[tool_serper, tool_binancenews],
    llm=llm,
    allow_delegation=True,
    max_iter=10
)

# creating a Researcher agent
researcher_agent = Agent(
    role="Senior Researcher",
    goal='Synthesize technical and fundamental analysis data to make informed trading decisions to day trading. Today is {Date}',
    verbose=True,
    memory=True,
    backstory=(
        """Provide the 1day market will be bullish or Bearish based on data to day traders. Driven by curiosity and a comprehensive understanding of both technical and
        fundamental analysis, you integrate various insights to provide well-rounded
        trading recommendations."""
    ),
    llm=llm
)
