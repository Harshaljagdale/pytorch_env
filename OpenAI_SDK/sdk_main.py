from agents import Agent, Runner
from dotenv import load_dotenv

load_dotenv()

agent = Agent(name="Assistant", instructions="You are a helpful assistant")

result = Runner.run_sync(agent, "Why sky is blue")
print(result.final_output)

# Code within the code,
# Functions calling themselves,
# Infinite loop's dance.