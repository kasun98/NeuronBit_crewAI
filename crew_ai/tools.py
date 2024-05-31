
from dotenv import load_dotenv
load_dotenv()
import os

os.environ['SERPER_API_KEY'] = os.getenv('SERPER_API_KEY')

from crewai_tools import SerperDevTool
from crewai_tools import ScrapeWebsiteTool
from crewai_tools import CSVSearchTool

# FA tools
tool_serper = SerperDevTool()
tool_investing = ScrapeWebsiteTool(website_url='https://www.investing.com/crypto/bitcoin/news')
tool_binancenews = ScrapeWebsiteTool(website_url='https://www.binance.com/en/square/news/bitcoin')

# TA tools
tool_cmc = ScrapeWebsiteTool(website_url='https://coinmarketcap.com/currencies/bitcoin/historical-data/')
tool_csv = CSVSearchTool(csv='../data/bitcoin.csv',
    config=dict(
        llm=dict(
            provider="google", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="gemini-1.5-flash",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="google", # or openai, ollama, ...
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )
)
#tool_csv = CSVSearchTool(csv='../data/processed_datav2.csv')


