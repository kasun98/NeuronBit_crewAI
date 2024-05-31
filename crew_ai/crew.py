from crewai import Crew,Process
from tasks import ta_task, fa_task, research_task
from agents import technical_analysis_agent, fundamental_analysis_agent, researcher_agent, llm
import datetime
from datetime import date, timedelta


crew=Crew(
    agents=[technical_analysis_agent, fundamental_analysis_agent, researcher_agent],
    tasks=[ta_task, fa_task, research_task],
    process=Process.hierarchical,
    manager_llm=llm,
    embedder={
        "provider": "google",
        "config":{
            "model": 'models/embedding-001',
            "task_type": "retrieval_document",
            "title": "Embeddings for Embedchain"}})


day = date.today()
l_day = day - timedelta(days=10)
result=crew.kickoff(inputs={'Date': day, 'L_date':l_day})
print(result)