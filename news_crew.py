import os
from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file


# 1. Ensure API keys are present in environment variables
# (Set them via export SERPER_API_KEY="your_key" and export OPENAI_API_KEY="your_key")
if not os.environ.get("SERPER_API_KEY"):
    raise ValueError("Error: SERPER_API_KEY environment variable is not set.")
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("Error: OPENAI_API_KEY environment variable is not set.")

# 2. Initialize the search tool for the Researcher
search_tool = SerperDevTool()

# 3. Create the two Agents with specific roles, goals, and backstories
researcher = Agent(
    role="Senior Tech Research Analyst",
    goal="Uncover and summarize today's top 3 most impactful technology news stories.",
    backstory=(
        "You are an expert tech researcher with a keen eye for disruptive trends "
        "and major industry breakthroughs. You scour the web daily to extract "
        "the most important headlines, facts, and URLs."
    ),
    tools=[search_tool],
    verbose=True,
    memory=False,
)

writer = Agent(
    role="Concise Tech Newsletter Editor",
    goal="Distill research findings into a punchy, exactly 5-line newsletter.",
    backstory=(
        "You are a master of brevity and clarity. Without relying on web searches, "
        "you take raw research notes and craft them into an ultra-short, engaging "
        "5-line newsletter format for busy tech professionals."
    ),
    tools=[],  # Writer cannot search at all
    verbose=True,
    memory=False,
)

# 4. Create the Tasks
# Task 1: Research today's top 3 tech stories
research_task = Task(
    description=(
        "Search the web for today's top 3 major tech news stories. "
        "Provide clear headlines, key details, and a concise summary for each."
    ),
    expected_output=(
        "A clean, structured markdown list of today's top 3 tech news stories, "
        "including the headline and 1-2 sentences of context for each."
    ),
    agent=researcher,
)

# Task 2: Transform the research into a 5-line newsletter
# We explicitly pass the research_task output via context=[research_task]
writing_task = Task(
    description=(
        "Using the research provided from the previous task, write today's tech newsletter. "
        "The newsletter must consist of exactly 5 lines (e.g., 1 title line and 4 bullet points, "
        "or 5 distinct sentences separated by line breaks)."
    ),
    expected_output=(
        "A newsletter consisting of exactly 5 lines of text (no more, no less), "
        "summarizing the top 3 tech stories."
    ),
    agent=writer,
    context=[research_task],  # Handing over the first task's output explicitly
)

# 5. Assemble the Crew with sequential processing
news_crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,
    verbose=True,
)

# 6. Kick off the crew execution
if __name__ == "__main__":
    print("## Starting Tech Newsletter Crew...")
    result = news_crew.kickoff()
    print("\n\n########################")
    print("## FINAL NEWSLETTER OUTPUT ##")
    print("########################\n")
    print(result)