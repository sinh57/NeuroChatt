"""
agent/tools.py
All LangChain tools for NeuroChat.

Each tool has full error handling — the agent never crashes on a bad tool call.
All external tools (Wikipedia, DuckDuckGo) have safe import fallbacks.
"""

import json
import math
import urllib.request
from datetime import datetime
from typing import List, Optional

# @tool decorator — stable across all LangChain versions
try:
    from langchain.tools import tool
except ImportError:
    from langchain_core.tools import tool  # type: ignore


# ── Calculator ────────────────────────────────────────────────────────────────
@tool
def calculator(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.
    Supports: +  -  *  /  **  sqrt  log  log10  sin  cos  tan  pi  e  abs  round  floor  ceil.
    Also handles percentages: '15% of 847'.
    Examples: '15% of 847', 'sqrt(144) + 50', '(3**4) / 9'
    """
    try:
        expr = expression.strip().lower()

        # Handle "X% of Y"
        if "%" in expr and "of" in expr:
            parts = expr.replace("%", "").split("of")
            pct = float(parts[0].strip())
            value = float(parts[1].strip())
            return f"{pct}% of {value} = {(pct / 100) * value:.4f}"

        expr = expr.replace("^", "**")
        safe_env = {
            "__builtins__": {},
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "abs": abs, "round": round, "pi": math.pi, "e": math.e,
            "pow": pow, "floor": math.floor, "ceil": math.ceil,
        }
        result = eval(expr, safe_env)  # noqa: S307
        return f"Result: {result}"
    except Exception as ex:
        return f"Calculator error: {ex}. Please check your expression."


# ── DateTime ──────────────────────────────────────────────────────────────────
@tool
def datetime_tool(query: str = "") -> str:
    """
    Return the current date, time, week number, and day of year.
    Use this for any question about today's date or the current time.
    """
    now = datetime.now()
    return (
        f"Current Date : {now.strftime('%A, %B %d, %Y')}\n"
        f"Current Time : {now.strftime('%I:%M:%S %p')}\n"
        f"Week Number  : {now.isocalendar()[1]}\n"
        f"Day of Year  : {now.timetuple().tm_yday}"
    )


# ── Weather ───────────────────────────────────────────────────────────────────
@tool
def weather_tool(city: str) -> str:
    """
    Fetch live weather for any city using the free wttr.in API (no API key needed).
    Input: city name — e.g. 'London', 'New York', 'Tokyo', 'Mumbai'
    """
    try:
        safe_city = city.strip().replace(" ", "+")
        url = f"https://wttr.in/{safe_city}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "neurochat/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        cur = data["current_condition"][0]
        return (
            f"Weather in {city}:\n"
            f"  Condition   : {cur['weatherDesc'][0]['value']}\n"
            f"  Temperature : {cur['temp_C']}°C / {cur['temp_F']}°F\n"
            f"  Feels Like  : {cur['FeelsLikeC']}°C\n"
            f"  Humidity    : {cur['humidity']}%\n"
            f"  Wind Speed  : {cur['windspeedKmph']} km/h"
        )
    except Exception as ex:
        return f"Could not fetch weather for '{city}'. Error: {ex}"


# ── Wikipedia ─────────────────────────────────────────────────────────────────
def _make_wikipedia_tool() -> Optional[object]:
    try:
        from langchain_community.tools import WikipediaQueryRun
        from langchain_community.utilities import WikipediaAPIWrapper
        wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=800)
        return WikipediaQueryRun(api_wrapper=wrapper)
    except Exception:
        return None


# ── Web Search (DuckDuckGo — no API key) ──────────────────────────────────────
def _make_search_tool() -> Optional[object]:
    # Try multiple import paths across langchain-community versions
    for path in [
        ("langchain_community.tools", "DuckDuckGoSearchRun"),
        ("langchain_community.tools.ddg_search.tool", "DuckDuckGoSearchRun"),
        ("langchain_community.tools.ddg_search", "DuckDuckGoSearchRun"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(path[0])
            cls = getattr(mod, path[1])
            return cls()
        except Exception:
            continue
    return None


# ── Tool registry ─────────────────────────────────────────────────────────────
def get_tools(selected: List[str]) -> List:
    """
    Return initialised tool objects for the given names.
    Any tool that fails to initialise is silently skipped (safe degradation).
    """
    registry = {
        "calculator": lambda: calculator,
        "datetime":   lambda: datetime_tool,
        "weather":    lambda: weather_tool,
        "wikipedia":  _make_wikipedia_tool,
        "web_search": _make_search_tool,
    }

    result = []
    for name in selected:
        if name in registry:
            t = registry[name]()
            if t is not None:
                result.append(t)
    return result
