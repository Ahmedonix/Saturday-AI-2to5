import os
import random
from dotenv import load_dotenv
import chainlit as cl
from typing import cast
from dataclasses import dataclass, field
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, function_tool
from agents.run import RunConfig

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# ----------------------------
# Context to track game state
# ----------------------------
@dataclass
class GameContext:
    player_name: str
    inventory: list[str] = field(default_factory=list)
    health: int = 100
    turn: int = 0

# ----------------------------
# Tools
# ----------------------------
@function_tool
def roll_dice() -> int:
    """Rolls a 6-sided dice and returns a number from 1 to 6."""
    return random.randint(1, 6)

@function_tool
def generate_event(scenario: str) -> str:
    """Generates a fantasy event based on the given scenario."""
    return f"As you {scenario}, a mysterious figure appears from the shadows..."

# ----------------------------
# Chainlit Chat Start
# ----------------------------
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

    narrator_agent = Agent[GameContext](
        name="NarratorAgent",
        instructions="Continue the fantasy story based on the player's input. Keep it immersive.",
        model=model,
    )

    monster_agent = Agent[GameContext](
        name="MonsterAgent",
        instructions="You control monster battles. Use roll_dice tool to simulate attacks. Report outcome and remaining health.",
        model=model,
        tools=[roll_dice],
    )

    item_agent = Agent[GameContext](
        name="ItemAgent",
        instructions="You manage loot and rewards. Add items to inventory and describe them.",
        model=model,
        tools=[generate_event],
    )

    game_master = Agent[GameContext](
        name="GameMasterAgent",
        instructions=(
            "You are the game master of a text-based adventure game. "
            "If the player encounters danger, hand off to MonsterAgent. "
            "If the player finds something, hand off to ItemAgent. Otherwise, continue the story using NarratorAgent."
        ),
        model=model,
        handoffs=[narrator_agent, monster_agent, item_agent],
    )

    cl.user_session.set("agent", game_master)
    cl.user_session.set("config", config)
    cl.user_session.set("chat_history", [])
    cl.user_session.set("game_context", GameContext(player_name="Hero"))

    await cl.Message(content="Welcome to the Fantasy Adventure Game! What will you do first?").send()

# ----------------------------
# On Message
# ----------------------------
@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content=" ")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))
    history = cl.user_session.get("chat_history") or []
    context: GameContext = cast(GameContext, cl.user_session.get("game_context"))

    context.turn += 1
    history.append({"role": "user", "content": message.content})

    try:
        result = Runner.run_streamed(
            starting_agent=agent,
            input=history,
            context=context,
            run_config=config,
        )

        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event.data, 'delta'):
                token = event.data.delta
                await msg.stream_token(token)

        response_content = result.final_output



        history.append({"role": "developer", "content": msg.content})
        cl.user_session.set("chat_history", history)

        print(f"Turn {context.turn} History: {history}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")
