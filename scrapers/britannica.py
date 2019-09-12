import requests
import pandas as pd

# getting all cities in the US and adding to key.csv

link = "https://www.britannica.com/topic/list-of-cities-and-towns-in-the-United-States-2023068"
f = requests.get(link)
text = f.text
# removing city names that are the same as state names
text = text.replace('<a href="https://www.britannica.com/place/Oregon-Illinois" class="md-crosslink">Oregon</a>', "")
text = text.replace('<a href="https://www.britannica.com/place/Washington-North-Carolina" class="md-crosslink">Washington</a>', "")
text = text.replace('<a href="https://www.britannica.com/place/Washington-Pennsylvania" class="md-crosslink">Washington</a>', "")
text = text.replace('<a href="https://www.britannica.com/place/Washington-Georgia" class="md-crosslink">Washington</a>', "")
text = text.replace('<a href="https://www.britannica.com/place/Virginia-Minnesota" class="md-crosslink">Virginia</a>', "")

cities = []

states = [
    "Alabama",
    "Alaska",
    "Arizona",
    "Arkansas",
    "California",
    "Colorado",
    "Connecticut",
    "Delaware",
    "Florida",
    "Georgia",
    "Hawaii",
    "Idaho",
    "Illinois",
    "Indiana",
    "Iowa",
    "Kansas",
    "Kentucky",
    "Louisiana",
    "Maine",
    "Maryland",
    "Massachusetts",
    "Michigan",
    "Minnesota",
    "Mississippi",
    "Missouri",
    "Montana",
    "Nebraska",
    "Nevada",
    "New Hampshire",
    "New Jersey",
    "New Mexico",
    "New York",
    "North Carolina",
    "North Dakota",
    "Ohio",
    "Oklahoma",
    "Oregon",
    "Pennsylvania",
    "Rhode Island",
    "South Carolina",
    "South Dakota",
    "Tennessee",
    "Texas",
    "Utah",
    "Vermont",
    "Virginia",
    "Washington",
    "West Virginia",
    "Wisconsin",
    "Wyoming"
    ]

for i in range(0, len(states)):
    if states[i] != "Wyoming":
        state_string = text[text.find("crosslink\">" + states[i] + "<"):text.find("crosslink\">" + states[i+1] + "<")]
    else:
        state_string = f.text[f.text.find("crosslink\">" + states[i]):]
    state_cities = state_string.split("com/place/")[1:]
    state_cities  = state_cities [:-1]
    state_cities  = [city[:city.find('"')] + "|" + states[i] for city in state_cities ]
    
    cities += state_cities

key = pd.read_csv("../key.csv")

# append empty rows to be filled
for i in range(0, len(cities)):
    key = key.append(pd.Series(), ignore_index=True)

# 48 because had some cities in already
old_stop = 48
key.loc[old_stop:,'city'] = [i.replace("-", " ") for i in cities]
key.loc[old_stop:,'state'] = [i[1] for i in key.loc[old_stop:,'city'].str.split("|")]
key.loc[old_stop:,'city'] = [i[0] for i in key.loc[old_stop:,'city'].str.split("|")]
# replace state names in city names
for i in range(old_stop, len(key)):
    key.loc[i, 'city'] = key.loc[i, 'city'].replace(" " + key.loc[i, 'state'], "")
    
# abbreviations
abbr = {
    "Alabama":"AL",
    "Alaska":"AK",
    "Arizona":"AZ",
    "Arkansas":"AR",
    "California":"CA",
    "Colorado":"CO",
    "Connecticut":"CT",
    "Delaware":"DE",
    "Florida":"FL",
    "Georgia":"GA",
    "Hawaii":"HI",
    "Idaho":"ID",
    "Illinois":"IL",
    "Indiana":"IN",
    "Iowa":"IA",
    "Kansas":"KS",
    "Kentucky":"KY",
    "Louisiana":"LA",
    "Maine":"ME",
    "Maryland":"MD",
    "Massachusetts":"MA",
    "Michigan":"MI",
    "Minnesota":"MN",
    "Mississippi":"MS",
    "Missouri":"MO",
    "Montana":"MT",
    "Nebraska":"NE",
    "Nevada":"NV",
    "New Hampshire":"NH",
    "New Jersey":"NJ",
    "New Mexico":"NM",
    "New York":"NY",
    "North Carolina":"NC",
    "North Dakota":"ND",
    "Ohio":"OH",
    "Oklahoma":"OK",
    "Oregon":"OR",
    "Pennsylvania":"PA",
    "Rhode Island":"RI",
    "South Carolina":"SC",
    "South Dakota":"SD",
    "Tennessee":"TN",
    "Texas":"TX",
    "Utah":"UT",
    "Vermont":"VT",
    "Virginia":"VA",
    "Washington":"WA",
    "West Virginia":"WV",
    "Wisconsin":"WI",
    "Wyoming":"WY"
}

key.loc[old_stop:,'state'] = key.loc[old_stop:,'state'].replace(abbr)

for i in range(old_stop, len(key)):
    key.loc[i, 'id'] = key.loc[i-1, 'id'] + 1
    
key.to_csv("../key.csv", index=False)