import os
import requests
import hashlib

def get_hash_based_weather(city_name):
    """
    Generate stable, realistic weather data based on the city name
    so that the demo is deterministic and behaves realistically without API keys.
    """
    city_lower = city_name.strip().lower()
    
    # Hash city name to get a stable seed
    h = int(hashlib.md5(city_lower.encode('utf-8')).hexdigest(), 16)
    
    # Pre-defined profiles for common agricultural cities
    if "miami" in city_lower or "orlando" in city_lower or "tropical" in city_lower:
        temp = 28.0 + (h % 50) / 10.0  # 28.0 to 33.0 °C
        humidity = 75 + (h % 15)       # 75% to 90%
        wind = 3.0 + (h % 40) / 10.0   # 3.0 to 7.0 m/s
        desc = "humid scattered clouds"
    elif "des moines" in city_lower or "chicago" in city_lower or "corn belt" in city_lower:
        temp = 20.0 + (h % 60) / 10.0  # 20.0 to 26.0 °C
        humidity = 55 + (h % 20)       # 55% to 75%
        wind = 4.0 + (h % 50) / 10.0   # 4.0 to 9.0 m/s
        desc = "clear sky and sunny"
    elif "seattle" in city_lower or "portland" in city_lower:
        temp = 14.0 + (h % 50) / 10.0  # 14.0 to 19.0 °C
        humidity = 80 + (h % 15)       # 80% to 95%
        wind = 2.0 + (h % 30) / 10.0   # 2.0 to 5.0 m/s
        desc = "light drizzle"
    else:
        # Default moderate farm weather
        temp = 18.0 + (h % 80) / 10.0  # 18.0 to 26.0 °C
        humidity = 50 + (h % 30)       # 50% to 80%
        wind = 2.0 + (h % 60) / 10.0   # 2.0 to 8.0 m/s
        conditions = ["clear sky", "few clouds", "scattered clouds", "broken clouds", "passing showers"]
        desc = conditions[h % len(conditions)]
        
    return {
        "city": city_name.strip().title(),
        "temperature": round(temp, 1),
        "humidity": int(humidity),
        "wind_speed": round(wind, 1),
        "description": desc,
        "is_mock": True
    }

def fetch_weather(city_name, api_key=None):
    """
    Fetch weather for a city. Tries the OpenWeather API if an API key is available.
    Falls back to a stable hash-based mock if the key is missing or the call fails.
    """
    if not city_name:
        return get_hash_based_weather("Greenhouse Lab")
        
    # Read key from env if not provided
    key = api_key or os.getenv("OPENWEATHER_API_KEY")
    
    if not key:
        return get_hash_based_weather(city_name)
        
    url = f"http://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city_name,
        "appid": key,
        "units": "metric"
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {
                "city": data.get("name", city_name),
                "temperature": round(data["main"]["temp"], 1),
                "humidity": int(data["main"]["humidity"]),
                "wind_speed": round(data["wind"]["speed"], 1),
                "description": data["weather"][0]["description"].lower(),
                "is_mock": False
            }
        else:
            print(f"OpenWeather API returned status code {response.status_code}. Falling back to mock weather.")
            return get_hash_based_weather(city_name)
    except Exception as e:
        print(f"Error querying OpenWeather API: {e}. Falling back to mock weather.")
        return get_hash_based_weather(city_name)
