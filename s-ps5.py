from collections import defaultdict

from pprint import pprint

data = {
    "Chicago": {"home": {"Q1": 1, "Q2": 2}, "computer": {"Q1": 1, "Q2": 2}},
    "NY": {"home": {"Q1": 1, "Q2": 2}, "computer": {"Q1": 1, "Q2": 2}},
    "Toronto": {"home": {"Q1": 1, "Q2": 2}, "computer": {"Q1": 1, "Q2": 2}},
}

countries = {"Chicago": "US", "NY": "US", "Toronto": "Canada"}

rolled_up_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for city, city_data in data.items():
    country = countries[city]
    existing_data = rolled_up_data[country]
    for product, time_data in city_data.items():
        for quarter, value in time_data.items():
            rolled_up_data[country][product][quarter] += value


pprint(dict(rolled_up_data))

diced_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for city, city_data in data.items():
    for product, time_data in city_data.items():
        for quarter, value in time_data.items():
            if city in ("Toronto") and quarter in ("Q1", "Q2") and product in ("home"):
                diced_data[city][product][quarter] = value

pprint(dict(diced_data))

sliced_data = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for city, city_data in data.items():
    for product, time_data in city_data.items():
        for quarter, value in time_data.items():
            if quarter in ("Q1"):
                sliced_data[city][product][quarter] = value
pprint(dict(sliced_data))