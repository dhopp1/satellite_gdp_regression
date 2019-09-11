# satellite\_gdp\_regression
A project to predict GDP and potentially other metrics of a city based on a satellite image of the city. <br><br>Satellite images come from google maps in the form of <https://www.google.com/maps/@lat,long,20000m/data=!3m1!1e3>, then screen shot, dimensions of 3360 × 1896.
# Key File
- `id` - unique id
- `country` - country where the city is located
- `region` - region of the city according to <https://en.wikipedia.org/wiki/United_Nations_geoscheme>
- `city` - the name of the city
- `latitude` - the latitude of the city (according to google maps)
- `longitude` - the longitude of the city (according to google maps)
- `population` - the population of the city
- `population_source` - the source of the population data
- `population_date` - the as of date of the population data
- `gdp` - nominal gdp of the city in USD
- `gdp_source` - source of the gdp data
- `gdp_date` - the as of date of the gdp date
- `age` - the founding year of the city
- `notes` - additional notes on the data