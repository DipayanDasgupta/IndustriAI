import requests
import pandas as pd

def fetch_world_bank_data(indicators, countries):
    records = []
    for indicator in indicators:
        for country in countries:
            url = f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}?format=json"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                if len(data) > 1 and data[1] is not None:
                    for item in data[1]:
                        if item.get('value') is not None:
                            records.append({
                                "Country": item['country']['value'],
                                "Country Code": country,
                                "Indicator": indicator,
                                "Year": item['date'],
                                "Value": item['value']
                            })
            else:
                print(f"Failed to fetch data for {indicator} in {country}. Status: {response.status_code}")
    return pd.DataFrame(records)

def fetch_project_data():
    # Example structure: You can replace this with actual data fetching
    data = [
        {"Project Name": "Solar Plant", "Budget": 500000, "Risk Score": 0.3, "ESG Metrics": "High renewable energy impact"},
        {"Project Name": "Waste Management", "Budget": 200000, "Risk Score": 0.5, "ESG Metrics": "Improves recycling efficiency"},
    ]
    return pd.DataFrame(data)

# Fetch and save World Bank data
indicators = ["SP.POP.TOTL", "NY.GDP.MKTP.CD"]
countries = ["IND", "USA", "CHN"]
esg_data = fetch_world_bank_data(indicators, countries)
if not esg_data.empty:
    esg_data.to_csv("../data/collected_data.csv", index=False)
    print("ESG data saved successfully.")
else:
    print("No ESG data fetched.")

# Fetch and save project data
project_data = fetch_project_data()
if not project_data.empty:
    project_data.to_csv("../data/projects.csv", index=False)
    print("Project data saved successfully.")
