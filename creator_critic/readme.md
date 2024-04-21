# Creator-Critic

Reflection
In the context of LLM agent building, reflection refers to the process of prompting an LLM to observe its past steps (along with potential observations from tools/the environment) to assess the quality of the chosen actions. This is then used downstream for things like re-planning, search, or evaluation.

Based on https://github.com/langchain-ai/langgraph/blob/main/examples/reflection/reflection.ipynb

![Reflection Schema](https://raw.githubusercontent.com/langchain-ai/langgraph/85f48da84ebb73adcbe0d46446ab2965e0daed87/examples/reflection/img/reflection.png)

Tools:
- LangChain
- LangGraph
- LangSmith
- Anthropic model

Files:
- [reflection_langgraph.py](reflection_langgraph.py) - code for graph creation
- [creator_critic_demo.ipynb](creator_critic_demo.ipynb) - example of applying graph to create description of GitHub repo.
