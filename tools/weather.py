from __future__ import annotations
import re
import requests
from langchain.tools import tool
from config import settings
from trace import run_tool_safely


def _normalize_location_query(location: str) -> str:
    """Normalize common city-name wording before geocoding."""
    value = location.strip()

    value = re.sub(
        r"\s+city$",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()

    return value


def _select_place(
    places: list[dict],
    original_location: str,
) -> dict:
    """Select the most plausible geocoding result."""
    if not places:
        raise ValueError(f"Location not found: {original_location}")

    normalized_query = _normalize_location_query(original_location).casefold()

    # Prefer an exact city-name match
    for place in places:
        name = str(place.get("name", "")).strip().casefold()

        if name == normalized_query:
            return place

    # Otherwise prefer a result whose name starts with the query
    for place in places:
        name = str(place.get("name", "")).strip().casefold()

        if name.startswith(normalized_query):
            return place

    # Fall back to the first OpenWeather result
    return places[0]


def _get_weather(
    location: str,
) -> str:
    if not settings.openweather_api_key:
        raise RuntimeError("OPENWEATHER_API_KEY is not configured in .env")

    geo_url = "https://api.openweathermap.org/" "geo/1.0/direct"

    normalized_location = _normalize_location_query(location)

    geo_resp = requests.get(
        geo_url,
        params={
            "q": normalized_location,
            "limit": 5,
            "appid": settings.openweather_api_key,
        },
        timeout=15,
    )

    geo_resp.raise_for_status()

    places = geo_resp.json()

    place = _select_place(
        places,
        location,
    )

    lat = place["lat"]
    lon = place["lon"]

    weather_resp = requests.get(
        "https://api.openweathermap.org/" "data/2.5/weather",
        params={
            "lat": lat,
            "lon": lon,
            "appid": settings.openweather_api_key,
            "units": "metric",
            "lang": "en",
        },
        timeout=15,
    )

    weather_resp.raise_for_status()

    data = weather_resp.json()

    description = data["weather"][0]["description"]

    temperature = data["main"]["temp"]
    feels_like = data["main"]["feels_like"]
    humidity = data["main"]["humidity"]

    wind_speed = data.get(
        "wind",
        {},
    ).get(
        "speed",
        0,
    )

    resolved = ", ".join(
        value
        for value in [
            place.get("name"),
            place.get("state"),
            place.get("country"),
        ]
        if value
    )

    return (
        f"Current weather for {resolved}: "
        f"{description}; "
        f"temperature {temperature:.1f}°C; "
        f"feels like {feels_like:.1f}°C; "
        f"humidity {humidity}%; "
        f"wind {wind_speed:.1f} m/s."
    )


@tool
def get_weather(
    location: str,
) -> str:
    """Get current weather for a city or location using live weather data.

    Use this tool when the user asks about current weather, temperature,
    humidity, wind, or similar real-time weather information.
    """
    return run_tool_safely("get_weather", _get_weather, location=location)
