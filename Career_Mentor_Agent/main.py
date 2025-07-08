import os
from dotenv import load_dotenv
import chainlit as cl
from typing import cast
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")





@cl.on_chat_start
async def start():

    external_client = AsyncOpenAI(
        api_key = gemini_api_key,
        base_url = base_url,
    )

    model = OpenAIChatCompletionsModel(
        model = "gemini-2.0-flash",
        openai_client = external_client,
    )

    config = RunConfig(
        model = model,
        model_provider = external_client,
        tracing_disabled = True,
    )

    get_career_roadmap = Agent(
        name = "Career Agent",
        instructions = "You will tell about skills needed for a career in a given field"
        "Answer by just telling the name of the skills needed",
        model = model,
    )

    skill_agent = Agent(
        name = "Skill Agent",
        instructions = "You will help people learn about the given skills",
        model = model,
    )

    job_agent = Agent(
        name = "Job Agent",
        instructions = "You tell people about the job roles according to their relevant skills",

        model = model,
    )

    agent = Agent(
        name = "Career Agent",
        instructions = "You will guide students through career"
        "If they ask about learning a skill, handoff to skill agent"
        "If they ask about jobs, handoff to job agent",
        tools = [get_career_roadmap.as_tool(
            tool_name = "get_career_roadmap",
            tool_description = "Give relevant skill set for the required field"
        )],
        handoffs = [skill_agent, job_agent]
    )
    cl.user_session.set("agent", agent)
    cl.user_session.set("config", config)
    cl.user_session.set("get_career_roadmap", get_career_roadmap)
    cl.user_session.set("skill_agent", skill_agent)
    cl.user_session.set("job_agent", job_agent)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="Welcome to the Career Assistant! How can I help you today?").send()

@cl.on_message
async def main(message: cl.Message):

    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    history = cl.user_session.get("chat_history") or []

    history.append({"role": "user", "content": message.content})

    try:
        result = await Runner.run(agent, history, run_config=config)

        response_content = result.final_output

        msg.content = response_content
        await msg.update()

        history.append({"role": "developer", "content": response_content})

        cl.user_session.set("chat_history", history)
        print(f"History: {history}")
    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")




