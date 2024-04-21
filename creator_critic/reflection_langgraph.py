from typing import List, Dict, TypedDict, Callable

from langchain.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph

ANTHROPIC_MODEL = 'claude-3-haiku-20240307'


def make_creator_chain(role: str, problem: str, result: str, verbose: bool = False) -> Callable[[], Dict[str, str]]:
    """
    Create a function that generates a high-level solution for a given problem.

    Args:
        role (str): The role of the person generating the solution.
        problem (str): The problem to be solved.
        result (str): The desired result of the solution.
        verbose (bool, optional): Whether to print the system prompt. Defaults to False.

    Returns:
        Callable[[], Dict[str, str]]: A function that, when called, returns a dictionary containing the high-level solution and the final result.
    """
    sys_prompt = (f"You are experienced {role}. You task is to write short (1-3 sentences) "
                  f"high-level description of possible solution for user's problem.")
    human_prompt_1 = f"""#### problem start
{problem}
#### problem end

Write short (1-3 sentences) high-level solution."""
    human_prompt_2 = f"Now {result}"
    model = ChatAnthropic(model_name=ANTHROPIC_MODEL, temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=sys_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model

    if verbose:
        print(prompt.format(messages=[]))

    def _invoke():
        _messages = [HumanMessage(content=human_prompt_1)]
        result_1 = chain.invoke({'messages': _messages}).content
        _messages.extend([AIMessage(content=result_1), HumanMessage(content=human_prompt_2)])
        result_2 = chain.invoke({'messages': _messages}).content
        return {"solution": result_1, 'result': result_2}

    return _invoke


def make_critic_chain(role: str, problem: str, result: str, verbose: bool = False) -> Callable[[Dict[str, str]], Dict[str, str]]:
    """
    Create a function that criticizes a proposed solution for a given problem and suggests an improved version.

    Args:
        role (str): The role of the person critiquing the solution.
        problem (str): The problem that the solution is intended to solve.
        result (str): The desired result of the solution.
        verbose (bool, optional): Whether to print the system prompt. Defaults to False.

    Returns:
        Callable[[Dict[str, str]], Dict[str, str]]: A function that, when called with a dictionary containing the proposed solution, returns a dictionary with the critique and an improved version of the solution.
    """
    sys_prompt = f"""You are experienced {role}. You task is to look through proposed solution for user's problem.
And then criticize this solution to find rooms for improvements.

#### problem start
{problem}
#### problem end
"""
    model = ChatAnthropic(model_name=ANTHROPIC_MODEL, temperature=0)
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=sys_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model

    if verbose:
        print(prompt.format(messages=[]))

    def _invoke(solution):
        human_prompt_1 = f"""#### solution start
{solution.get('result')}
#### solution end

Write 1-3 sentences about how this solution can be improved."""
        _messages = [HumanMessage(content=human_prompt_1)]
        result_1 = chain.invoke({'messages': _messages}).content

        human_prompt_2 = f"Now propose improved version of required result: {result}"
        _messages.extend([AIMessage(content=result_1), HumanMessage(content=human_prompt_2)])
        result_2 = chain.invoke({'messages': _messages}).content
        return {"critic": result_1, "result": result_2}

    return _invoke


def print_sol(solution_dict):
    print(f"Solution:\n{solution_dict.get('solution', 'no solution')}\n-------------------------------------\n"
          f"Result:\n{solution_dict.get('result', 'no result')}")


def print_critic_sol(solution_dict):
    print(f"Critic:\n{solution_dict.get('critic', 'no critic')}\n-------------------------------------\n"
          f"Improved Result:\n{solution_dict.get('result', 'no result')}")


class GraphState(TypedDict):
    messages: List[BaseMessage]
    init_params: Dict
    flow_params: Dict
    solutions: List[Dict]


def init_state(role, problem, required_result, max_iterations=1):
    return {
        'messages': [],
        'init_params': {'role': role, 'problem': problem, 'result': required_result},
        'flow_params': {'iterations_done': 0, 'max_iterations': max_iterations},
        'solutions': [],
    }


def build_graph() -> StateGraph:
    """
    Build a state graph for the solution generation and critique process.

    Returns:
        StateGraph: The compiled state graph.
    """
    def creator_node(state: GraphState) -> GraphState:
        creator = make_creator_chain(
            role=state['init_params']['role'],
            problem=state['init_params']['problem'],
            result=state['init_params']['result'],
        )
        _solution = creator()
        state['solutions'].append(_solution)
        return state

    def critic_node(state: GraphState) -> GraphState:
        critic = make_critic_chain(
            role=state['init_params']['role'],
            problem=state['init_params']['problem'],
            result=state['init_params']['result'],
        )
        _prev_solution = state['solutions'][-1]
        _solution = critic(_prev_solution)
        state['solutions'].append(_solution)
        state['flow_params']['iterations_done'] += 1
        return state

    def should_continue(state: GraphState) -> str:
        if state['flow_params']['iterations_done'] >= state['flow_params']['max_iterations']:
            return END
        return "reflect"

    builder = StateGraph(GraphState)
    builder.add_node("generate", creator_node)
    builder.add_node("reflect", critic_node)
    builder.set_entry_point("generate")
    builder.add_edge("generate", "reflect")

    builder.add_conditional_edges("reflect", should_continue)
    graph = builder.compile()
    return graph


def get_last_solution(state: GraphState) -> Dict:
    return state['solutions'][-1]
