from crewai import Task
from tools import tool_serper, tool_investing, tool_binancenews, tool_cmc, tool_csv
from agents import technical_analysis_agent, fundamental_analysis_agent, researcher_agent

'''
# Research task
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points,"
    "its market opportunities, and potential risks."
  ),
  expected_output='A comprehensive 3 paragraphs long report on the latest AI trends.',
  tools=[tool],
  agent=news_researcher,
)

# Writing task with language model configuration
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging, and positive."
  ),
  expected_output='A 4 paragraph article on {topic} advancements formatted as markdown.',
  tools=[tool],
  agent=news_writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
)
'''

# Technical Analysis Task
ta_task = Task(
    description=(
        """Analyze the Bitcoin price movements within {L_date} and {Date} using
        technical indicators, in bitcoin.csv csv file you have past 10 days bitcoin price data like btc_price = Bitcoin price, and use serper tool to search on web. Today is {Date}. Focus on identifying key
        trends and provide a forecast for whether the market will be bullish
        or bearish over the next 24 hours."""
        ),
        expected_output='A detailed report with technical analysis, including charts and a bullish/bearish forecast for the next 24 hours.',
        tools=[tool_serper, tool_cmc, tool_csv],
        agent=technical_analysis_agent)

# Fundamental Analysis Task
fa_task = Task(
    description=(
        """Analyze Bitcoin price movements using fundamental economic factors and
        news from the last 24 hours. Today is {Date}. Focus on identifying key events and
        factors that could influence the market direction. Provide a forecast
        for whether the market will be bullish or bearish over the next 24 hours."""
        ),
        expected_output='A comprehensive report on fundamental analysis, including key economic indicators and news summaries with a bullish/bearish forecast for the next 24 hours.',
        tools=[tool_serper, tool_binancenews],
        agent=fundamental_analysis_agent)

# Research Task
research_task = Task(
    description=(
        """Synthesize the data from the technical and fundamental analysis
        reports to provide an overall market forecast for Bitcoin over the
        next 24 hours. Today is {Date}.  Focus on integrating insights from both analyses and
        clearly articulating a well-rounded trading recommendation."""
        ),
        expected_output='An integrated report combining technical and fundamental analysis with a clear bullish/bearish forecast and trading recommendation for the next 24 hours.',
        agent=researcher_agent,
        context=[ta_task, fa_task],
        output_file='{Date}.md')
