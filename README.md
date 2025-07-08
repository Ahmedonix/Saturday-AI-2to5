# 🔮 Multi-Agent AI Projects using OpenAI Agents SDK + Chainlit

This repository contains 3 intelligent agent-powered projects built using the **OpenAI Agents SDK**, **Gemini via OpenAI-compatible API**, and **Chainlit** for interactive chat interfaces.

---

## 🚀 Projects Overview

### 1. 🧑‍🎓 Career Mentor Agent

A multi-agent system that helps students explore career paths.

**Features:**
- Recommends careers based on interests
- Shares roadmaps of skills needed using a tool: `get_career_roadmap()`
- Agent handoffs for deeper guidance:
  - **CareerAgent** → suggests fields
  - **SkillAgent** → shows how to learn them
  - **JobAgent** → lists real-world job roles

**Stack:**
- Chainlit UI
- OpenAI Agents SDK
- Gemini API via OpenAI-compatible client
- Tool + Handoff based architecture

---

### 2. 🧳 AI Travel Designer Agent

An AI-powered travel planner that builds full trips based on your mood or interests.

**Features:**
- Suggests destinations based on user mood
- Books flights and hotels using tools:
  - `get_flights(destination)`
  - `suggest_hotels(destination)`
- Agent handoffs:
  - **DestinationAgent** → finds matching places
  - **BookingAgent** → shows options
  - **ExploreAgent** → recommends attractions and food

**Stack:**
- Chainlit for real-time chat
- OpenAI Agents SDK
- Gemini model (OpenAI-compatible)
- Mock data for hotel and flight suggestions

---

### 3. 🧙‍♂️ Game Master Agent (Fantasy Adventure Game)

An immersive, text-based roleplaying game powered by agents.

**Features:**
- Narrates fantasy stories based on player input
- Uses tools:
  - `roll_dice()` → for combat outcomes
  - `generate_event()` → for story progression
- Multi-agent structure:
  - **NarratorAgent** → tells the story
  - **MonsterAgent** → controls battles
  - **ItemAgent** → handles inventory & loot
- Central **GameMasterAgent** controls flow and hands off responsibilities

**Stack:**
- Chainlit frontend
- OpenAI Agents SDK
- Gemini-compatible LLM
- In-game context for health, inventory, and story turns

---

## 🧰 Setup Instructions

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
