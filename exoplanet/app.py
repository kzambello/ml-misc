import numpy as np
import pandas as pd

import re
from pathlib import Path

from datetime import datetime
import time

from anyascii import anyascii

from typing import Any, Literal

from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import before_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.messages import RemoveMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.runtime import Runtime
from langchain.tools import tool, ToolRuntime
from langchain_aws import ChatBedrockConverse

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

app = FastAPI()


def load_system_prompt() -> str:
    """Load system prompt from external file."""
    prompt_path = Path(__file__).parent / "prompts" / "system_prompt.txt"
    try:
        return prompt_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise RuntimeError(
            f"System prompt file not found at {prompt_path}. "
            "Please ensure prompts/system_prompt.txt exists."
        )


def remove_accents(old):
    """Removes common accent characters."""

    new = re.sub(r"[àá]", "a`", old)
    new = re.sub(r"[èé]", "e`", new)
    new = re.sub(r"[ìí]", "i`", new)
    new = re.sub(r"[òó]", "o`", new)
    new = re.sub(r"[ùú]", "u`", new)

    new = re.sub(r"[âãäå]", "a", new)
    new = re.sub(r"[êë]", "e", new)
    new = re.sub(r"[îï]", "i", new)
    new = re.sub(r"[ôö]", "o", new)
    new = re.sub(r"[ûü]", "u", new)

    new = re.sub(r"[ÀÁ]", "A`", new)
    new = re.sub(r"[ÈÉ]", "E`", new)
    new = re.sub(r"[ÌÍ]", "I`", new)
    new = re.sub(r"[ÒÓ]", "O`", new)
    new = re.sub(r"[ÙÚ]", "U`", new)

    new = anyascii(new)

    return new


df = pd.read_csv("data/exoplanets.csv", index_col=0)

df["pl_name"] = df["pl_name"].apply(lambda x: x.replace(" ", "-"))
df["hostname"] = df["hostname"].apply(lambda x: x.replace(" ", "-"))
df["pl_name"] = df["pl_name"].astype(str)
df["hostname"] = df["hostname"].astype(str)


@tool
def list_columns() -> str:
    """List all available columns in the exoplanet dataset with descriptions."""

    columns_info = {
        "pl_name": "Exoplanet name",
        "hostname": "Host star name",
        "disc_year": "Year of discovery",
        "pl_orbper": "Orbital period (days)",
        "pl_rade": "Exoplanet radius (Earth radii)",
        "pl_bmasse": "Exoplanet mass (Earth masses)",
        "pl_eqt": "Exoplanet temperature (K)",
        "st_teff": "Host star temperature (K)",
        "st_rad": "Host star radius (Solar radii)",
        "st_mass": "Host star mass (Solar masses)",
    }

    result = "Available columns:\n"
    for col, desc in columns_info.items():
        result += f"  * {col}: {desc}\n"

    return result


@tool
def find_min(column: str, filters: dict = None) -> str:
    """Find the minimum value of a column and return the exoplanet(s) with it.

    Args:
        column: Column name (e.g., 'disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt')
        filters: Optional dict of {column: (min, max)} or {column: exact_value} for filtering first

    Returns:
        String with the result and exoplanet name(s)
    """

    df_copy = df.copy()

    # apply filters if provided
    if filters:
        for col, value in filters.items():
            if col not in df.columns:
                return f"Error: Column '{col}' not found"

            # Normalize value
            if isinstance(value, list):
                if len(value) == 2:
                    value = tuple(value)  # Convert [min, max] to (min, max)
                elif len(value) == 1:
                    value = value[0]  # Unwrap single-element list
            if isinstance(value, tuple) and len(value) == 2:
                min_val, max_val = value
                df_copy = df_copy[(df_copy[col] >= min_val) & (df_copy[col] <= max_val)]
            else:
                df_copy = df_copy[df_copy[col] == value]

    min_val = df_copy[column].min()
    min_rows = df_copy[df_copy[column] == min_val]

    result = f"Minimum {column}: {min_val}\n"
    result += f"Exoplanet(s): {', '.join(min_rows['pl_name'].values)}"
    return result


@tool
def find_max(column: str, filters: dict = None) -> str:
    """Find the maximum value of a column and return the exoplanet(s) with it.

    Args:
        column: Column name (e.g., 'disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt')
        filters: Optional dict of {column: (min, max)} or {column: exact_value} for filtering first

    Returns:
        String with the result and exoplanet name(s)
    """

    df_copy = df.copy()

    # apply filters if provided
    if filters:
        for col, value in filters.items():
            if col not in df.columns:
                return f"Error: Column '{col}' not found"

            # Normalize value
            if isinstance(value, list):
                if len(value) == 2:
                    value = tuple(value)  # Convert [min, max] to (min, max)
                elif len(value) == 1:
                    value = value[0]  # Unwrap single-element list
            if isinstance(value, tuple) and len(value) == 2:
                min_val, max_val = value
                df_copy = df_copy[(df_copy[col] >= min_val) & (df_copy[col] <= max_val)]
            else:
                df_copy = df_copy[df_copy[col] == value]

    max_val = df_copy[column].max()
    max_rows = df_copy[df_copy[column] == max_val]

    result = f"Maximum {column}: {max_val}\n"
    result += f"Exoplanet(s): {', '.join(max_rows['pl_name'].values)}"
    return result


@tool
def calculate_stats(column: str, filters: dict = None) -> str:
    """Calculate average and standard deviation of a column.

    Args:
        column: Column name (e.g., 'disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt')
        filters: Optional dict of {column: (min, max)} or {column: exact_value} for filtering first

    Returns:
        String with mean and standard deviation values
    """

    df_copy = df.copy()

    # apply filters if provided
    if filters:
        for col, value in filters.items():
            if col not in df.columns:
                return f"Error: Column '{col}' not found"

            # Normalize value
            if isinstance(value, list):
                if len(value) == 2:
                    value = tuple(value)  # Convert [min, max] to (min, max)
                elif len(value) == 1:
                    value = value[0]  # Unwrap single-element list
            if isinstance(value, tuple) and len(value) == 2:
                min_val, max_val = value
                df_copy = df_copy[(df_copy[col] >= min_val) & (df_copy[col] <= max_val)]
            else:
                df_copy = df_copy[df_copy[col] == value]

    mean_val = df_copy[column].mean()
    std_val = df_copy[column].std()

    result = f"Average {column}: {mean_val:.2f}\n"
    result += f"Standard deviation: {std_val:.2f}"
    return result


@tool
def filter_range(column: str, min_value: float | int, max_value: float | int) -> str:
    """Filter exoplanets by a property range.

    Args:
        column: Column name (e.g., 'disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt')
        min_value: Lower bound (inclusive)
        max_value: Upper bound (inclusive)

    Returns:
        String with count and sample of exoplanets in property range
    """

    df_copy = df.copy()

    df_match = df_copy[(df_copy[column] >= min_value) & (df_copy[column] <= max_value)]

    if df_match.empty:
        return f"No exoplanet found with property '{column}' between {min_value} and {max_value}."

    result_list = []
    for idx, row in df_match.iterrows():
        result_list.append(
            f" * Exoplanet name: {row['pl_name']}, Host star name: {row['hostname']}, Year of discovery: {row['disc_year']}, Orbital period: {row['pl_orbper']} days, Exoplanet radius: {row['pl_rade']} Earth radii, Exoplanet mass: {row['pl_bmasse']} Earth masses, Exoplanet temperature: {row['pl_eqt']} K, Host star temperature: {row['st_teff']} K, Host star radius: {row['st_rad']} Solar radii, Host star mass: {row['st_mass']} Solar masses"
        )

    count = len(result_list)
    sample = result_list[0:5]

    result = f"Found {count} exoplanet(s) with '{column}' between {min_value} and {max_value}.\n"
    result += "Sample: \n" + "\n".join(sample)
    if count > 5:
        result += f"\n ... and {count - 5} more"
    return result


@tool
def filter_exact_value(column: str, value: str | float | int) -> str:
    """
    Find exoplanet(s) by property value.

    Args:
        column: Column name (e.g., 'pl_name', 'hostname', 'disc_year', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt')
        value: exact value

    Returns:
        String with matching exoplanet(s) details

    """

    df_copy = df.copy()

    df_match = df_copy[df_copy[column] == value]

    if df_match.empty:
        return f"No exoplanet found with property '{column}' = {value}."

    result_list = []
    for idx, row in df_match.iterrows():
        result_list.append(
            f" * Exoplanet name: {row['pl_name']}, Host star name: {row['hostname']}, Year of discovery: {row['disc_year']}, Orbital period: {row['pl_orbper']} days, Exoplanet radius: {row['pl_rade']} Earth radii, Exoplanet mass: {row['pl_bmasse']} Earth masses, Exoplanet temperature: {row['pl_eqt']} K, Host star temperature: {row['st_teff']} K, Host star radius: {row['st_rad']} Solar radii, Host star mass: {row['st_mass']} Solar masses"
        )

    count = len(result_list)
    sample = result_list[0:5]

    result = f"Found {count} exoplanet(s) with '{column}' = {value}.\n"
    result += "Sample: \n" + "\n".join(sample)
    if count > 5:
        result += f"\n ... and {count - 5} more"
    return result


tools = [
    list_columns,
    find_min,
    find_max,
    calculate_stats,
    filter_range,
    filter_exact_value,
]

model = ChatBedrockConverse(
    model="nvidia.nemotron-super-3-120b",
    region_name="us-east-1",
    max_tokens=128,
    temperature=0.20,
    top_p=1.00,
).bind_tools(tools)

system_prompt = load_system_prompt()


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Keep only the last few messages to fit context window."""
    messages = state["messages"]

    limit = 16  # keep this even

    if len(messages) <= limit + 1:
        return None

    first_msg = messages[0]  # system prompt

    body_messages = messages[1:]

    if len(body_messages) % 2 == 0:
        recent_messages = body_messages[-limit:]
    else:
        recent_messages = body_messages[-limit + 1 :]

    new_messages = [first_msg] + recent_messages

    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *new_messages]}


agent = create_agent(
    model=model,
    tools=tools,
    system_prompt=system_prompt,
    middleware=[trim_messages],
    checkpointer=InMemorySaver(),
)

thread_config = {"configurable": {"thread_id": "1"}}


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str


class RegisterRequest(BaseModel):
    query: str
    sender: Literal["human", "ai"]


class RegisterResponse(BaseModel):
    answer: str


@app.post("/query")
def handle_query(request: QueryRequest) -> QueryResponse:

    # GET INITIAL MESSAGE COUNT
    state_snapshot = agent.get_state(thread_config)
    initial_count = (
        len(state_snapshot.values.get("messages", [])) if state_snapshot else 0
    )
    #

    # START TIMING HERE
    start = time.time()
    print(f"[{datetime.now()}] Starting inference...")
    #

    # INVOKE AGENT
    response = agent.invoke(
        {"messages": [{"role": "user", "content": request.query}]}, thread_config
    )
    #

    # END TIMING HERE
    elapsed = time.time() - start
    print(f"[{datetime.now()}] Inference completed in {elapsed:.1f}s")
    output_tokens = len(response["messages"][-1].content.split())
    print(
        f"[{datetime.now()}] Output length: ~{output_tokens} words (~{output_tokens//0.75} tokens)"
    )
    #

    # PRINT TOOLS USED
    tools_used = []
    for msg in response["messages"][initial_count:]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                name = (
                    tool_call.get("name")
                    if isinstance(tool_call, dict)
                    else getattr(tool_call, "name", None)
                )
                if name:
                    tools_used.append(name)
        elif isinstance(msg, dict) and "tool_calls" in msg:
            for tool_call in msg["tool_calls"]:
                name = tool_call.get("name")
                if name:
                    tools_used.append(name)
    if tools_used:
        print(f"[{datetime.now()}] Tools used: {tools_used}")
    else:
        print(f"[{datetime.now()}] Tools used: None")
    #

    # EXTRACT THE FINAL MESSAGE
    if isinstance(response, dict) and "messages" in response:
        final_message = response["messages"][-1]
        answer = final_message.content
        answer = remove_accents(answer)
    else:
        answer = str(response)
    #

    return QueryResponse(answer=answer)


@app.post("/register")
def handle_register(request: RegisterRequest) -> RegisterResponse:
    user_text = request.query

    if request.sender == "ai":
        message = AIMessage(content=user_text)
    else:
        message = HumanMessage(content=user_text)

    agent.update_state(thread_config, {"messages": [message]})

    return RegisterResponse(answer="")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8010, timeout_keep_alive=90)
