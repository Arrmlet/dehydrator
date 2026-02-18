from dehydrator._tokenizer import tokenize_query, tokenize_tool


def test_snake_case_name():
    tool = {"name": "get_weather_forecast", "description": "", "input_schema": {}}
    tokens = tokenize_tool(tool)
    assert "get" in tokens
    assert "weather" in tokens
    assert "forecast" in tokens


def test_camel_case_name():
    tool = {"name": "getWeatherForecast", "description": "", "input_schema": {}}
    tokens = tokenize_tool(tool)
    assert "get" in tokens
    assert "weather" in tokens
    assert "forecast" in tokens


def test_description_tokenized():
    tool = {
        "name": "tool",
        "description": "Fetches the current weather for a city",
        "input_schema": {},
    }
    tokens = tokenize_tool(tool)
    assert "current" in tokens
    assert "weather" in tokens
    assert "city" in tokens


def test_schema_properties():
    tool = {
        "name": "tool",
        "description": "",
        "input_schema": {
            "type": "object",
            "properties": {
                "city_name": {
                    "type": "string",
                    "description": "The name of the city",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
        },
    }
    tokens = tokenize_tool(tool)
    assert "city" in tokens
    assert "name" in tokens
    assert "celsius" in tokens
    assert "fahrenheit" in tokens


def test_nested_object_schema():
    tool = {
        "name": "tool",
        "description": "",
        "input_schema": {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "zip_code": {
                            "type": "string",
                            "description": "Postal code",
                        },
                    },
                },
            },
        },
    }
    tokens = tokenize_tool(tool)
    assert "zip" in tokens
    assert "code" in tokens
    assert "postal" in tokens


def test_tokenize_query():
    tokens = tokenize_query("send an email to someone")
    assert tokens == ["send", "an", "email", "to", "someone"]


def test_empty_tool():
    tool = {"name": "", "description": "", "input_schema": {}}
    tokens = tokenize_tool(tool)
    assert tokens == []


def test_mcp_format_input_schema():
    """Tools using MCP camelCase inputSchema are tokenized correctly."""
    tool = {
        "name": "get_weather",
        "description": "Get weather forecast",
        "inputSchema": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
            },
        },
    }
    tokens = tokenize_tool(tool)
    assert "get" in tokens
    assert "weather" in tokens
    assert "city" in tokens
