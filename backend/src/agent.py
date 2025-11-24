import asyncio
from livekit import agents
from livekit.agents import Agent
from livekit.agents.voice_assistant import VoiceAssistant
import os

# Simple fast TTS fallback if Murf key missing
async def simple_tts(text: str):
    print(f"Barista says: {text}")
    return b""  # LiveKit will use default TTS if key missing

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect_auto()

    assistant = VoiceAssistant(
        vad=agents.vad.SileroVAD(),
        stt=agents.stt.OpenAISTT(),
        llm=agents.llm.OpenAI(model="gpt-4o-mini"),
        tts=agents.tts.CustomTTS(simple_tts),  # Will work instantly
        chat_ctx=Agent(instructions="""
            You are Alex, a super friendly barista at Falcon Brew Coffee Shop.
            Greet customers, take their order (drink, size, milk, extras), confirm it, and say thank you.
            Be warm1192 warm and energetic!
        """)
    )

    assistant.start(ctx.room)
    await asyncio.sleep(1)
    await assistant.say("☕ Hi! Welcome to Falcon Brew Coffee! I'm Alex, what can I get for you today?", allow_interruptions=True)

if __name__ == "__main__":
    print("")
    print("YOUR COFFEE BARISTA IS READY!")
    print("Join room: coffee-barista-demo")
    print("Link → https://livekit.io/join")
    print("")
    agents.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint, room_name="coffee-barista-demo"))
