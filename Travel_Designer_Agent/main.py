import os
from dotenv import load_dotenv
import chainlit as cl
from typing import cast, List
from dataclasses import dataclass
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig


load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

    
@dataclass
class UserTravelContext:
    user_id: str
    mood: str = ""
    destination: str = ""

@function_tool
def get_flights(destination: str) -> str:
    return f"Flights available to {destination}: Flight A (10am), Flight B (3pm), Flight C (9pm)"

@function_tool
def suggest_hotels(destination: str) -> List[str]:
    hotels = {
        "paris": ["Hotel Lumière", "Eiffel Stay", "Champs Boutique Hotel"],
        "tokyo": ["Shinjuku Inn", "Tokyo Grand Palace", "Nihon Comfort Hotel"],
        "new york": ["Manhattan Suites", "Central Park Hotel", "Liberty Stay"]
    }
    return hotels.get(destination.lower(), ["No hotel data available."])


@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=gemini_api_key,
        base_url=base_url,
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.0-flash",
        openai_client=external_client,
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True,
    )

    destination_agent = Agent[UserTravelContext](
        name="DestinationAgent",
        instructions="Ask the user about their mood or interests and suggest a matching travel destination. Always store the destination in context.",
        model=model,
    )

    booking_agent = Agent[UserTravelContext](
        name="BookingAgent",
        instructions="Use get_flights and suggest_hotels tools to simulate booking a trip for the selected destination. Show flights and hotels.",
        model=model,
        tools=[get_flights, suggest_hotels],
    )

    explore_agent = Agent[UserTravelContext](
        name="ExploreAgent",
        instructions="Recommend top attractions, foods, and experiences for the destination in context.",
        model=model,
    )

    travel_agent = Agent[UserTravelContext](
        name="TravelAgent",
        instructions="You are a helpful travel assistant. Use DestinationAgent to suggest destinations based on mood, BookingAgent to book flights and hotels, and ExploreAgent to suggest attractions and food.",
        model=model,
        handoffs=[destination_agent, booking_agent, explore_agent],
    )

    cl.user_session.set("agent", travel_agent)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("user_context", UserTravelContext(user_id="Ahmed"))

    await cl.Message(content="Welcome to the AI Travel Designer! Where would you like to go today?").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="Planning your travel experience...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get("chat_history") or []
    context: UserTravelContext = cast(UserTravelContext, cl.user_session.get("user_context"))

    history.append({"role": "user", "content": message.content})

    try:
        result = await Runner.run(
            starting_agent=agent,
            input=history,
            context=context,
            run_config=config,
        )

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