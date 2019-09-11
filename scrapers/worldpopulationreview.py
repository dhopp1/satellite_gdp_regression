import requests
import pandas as pd
import time

def get_coord(link):
    """
    function to return the latitude and longitude from a worldpopulation review URL
    
    parameters:
    ----------
    link : str
        the url
        
    returns:
    --------
    str
        the latitude and longitude of the url
    """
    f = requests.get(link)
    start = f.text.find("maps/?q=") + 8
    end = f.text[start:].find('"')
    return f.text[start:start+end]


def get_pop(link):
    """
    function to return the population from a worldpopulation review URL
    
    parameters:
    ----------
    link : str
        the url
        
    returns:
    --------
    int
        the population
    """
    f = requests.get(link)
    global_start = f.text.find("How Many People Live in")
    start = f.text[global_start:].find("30px") + 6
    end = f.text[global_start:].find("</span>")
    return int(f.text[global_start+start:global_start+end].replace(",", ""))

def get_income(link):
    """
    function to return the mean household income from a worldpopulation review URL
    
    parameters:
    ----------
    link : str
        the url
        
    returns:
    --------
    int
        the mean household income
    """
    f = requests.get(link)
    global_start = f.text.find("while the mean household income is")
    start = f.text[global_start:].find("<!-- -->") + 8
    end = f.text[global_start:].find("</span>")
    return int(f.text[global_start+start:global_start+end].replace(",", ""))

key = pd.read_csv("../key.csv")
urls = "http://worldpopulationreview.com/us-cities/" + key.city.str.lower().str.replace(" ", "-") + "-" + key.state.str.lower() + "-population"

for i in range(0, len(urls)):
    if True:#pd.isna(key.loc[i, "population"]) | pd.isna(key.loc[i, "latitude"]) | pd.isna(key.loc[i, "gdp"]):
        try:
            key.loc[i, "population"] = get_pop(urls[i])
            key.loc[i, "population_date"] = 2018

            coord = get_coord(urls[i])
            key.loc[i, "latitude"] = float(coord.split(",")[0])
            key.loc[i, "longitude"] = float(coord.split(",")[1])
            
            key.loc[i, "gdp"] = get_income(urls[i]) * key.loc[i, "population"] / 1000000
            key.loc[i, "population_date"] = 2018
            
            time.sleep(2)
        except:
            pass
    print(i)

key['population_source'] = urls
key['gdp_source'] = urls

key.to_csv("../key.csv", index=False)
    
