### MIGUEL GONZALEZ
# Import requests

from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import datetime

### HTML TABLE

```
def date_time(table_cells):
    return [data_time.strip() for data_time in list(table_cells.strings)][0:2]

def booster_version(table_cells):
    out=''.join([booster_version for i,booster_version in enumerate( table_cells.strings) if i%2==0][0:-1])
    return out

def landing_status(table_cells):
    out=[i for i in table_cells.strings][0]
    return out

def get_mass(table_cells):
    mass=unicodedata.normalize("NFKD", table_cells.text).strip()
    if mass:
        mass.find("kg")
        new_mass=mass[0:mass.find("kg")+2]
    else:
        new_mass=0
    return new_mass

def extract_column_from_header(row):
    if (row.br):
        row.br.extract()
    if row.a:
        row.a.extract()
    if row.sup:
        row.sup.extract()    
    colunm_name = ' '.join(row.contents)
    # Filter the digit and empty names
    if not(colunm_name.strip().isdigit()):
        colunm_name = colunm_name.strip()
        return colunm_name    
        
```

website = 'https://en.wikipedia.org/wiki/List_of_Falcon_9_and_Falcon_Heavy_launches?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMDS0321ENSkillsNetwork26802033-2022-01-01'

```
page = requests.get(website)
soup = BeautifulSoup(page.content, 'html.parser')
soup.tittle
rows = soup.findAll('table',"wikitable plainrowheaders collapsible")
rows[2].find('tbody').findAll('tr')

rows[0].find_all('td')[1].get_text
rows[0].find_all('td')[2].get_text

names = [] 
launch_dict= dict.fromkeys(names)
for row in rows:
  launch_dict['Flight No.'] = []
  launch_dict['Launch site'] = []
  launch_dict['Payload'] = []
  launch_dict['Payload mass'] = []
  launch_dict['Orbit'] = []
  launch_dict['Customer'] = []
  launch_dict['Launch outcome'] = []
  launch_dict['Version Booster']=[]
  launch_dict['Booster landing']=[]
  launch_dict['Date']=[]
  launch_dict['Time']=[]

df=pd.DataFrame(launch_dict)
df.to_csv('spacex_web_scraped_part_1.csv', index=False)
```

## Load Data From CSV File
``` 
df = pd.read_csv('space_x_web_scraped_part_1.csv')
df.head() 
```

## Perform exploratory Data Analysis and determine Training Labels
* Exploratory Data Analysis
* Determine Training Labels

``` 
df=pd.read_csv("space_x_web_scraped_part_1")
df.head(10) 
df.isnull().sum()/df.count()*100
df.dtypes
#Use the method value_counts() on the column LaunchSite to determine the number of launches on each site:
df['LaunchSite'].value_counts()

``` 

##Each launch aims to an dedicated orbit, and here are some common orbit types:

* LEO: Low Earth orbit (LEO)is an Earth-centred orbit with an altitude of 2,000 km (1,200 mi) or less (approximately one-third of the radius of Earth),[1] or with at least 11.25 periods per day (an orbital period of 128 minutes or less) and an eccentricity less than 0.25.[2] Most of the manmade objects in outer space are in LEO [1].

* VLEO: Very Low Earth Orbits (VLEO) can be defined as the orbits with a mean altitude below 450 km. Operating in these orbits can provide a number of benefits to Earth observation spacecraft as the spacecraft operates closer to the observation[2].

* GTO A geosynchronous orbit is a high Earth orbit that allows satellites to match Earth's rotation. Located at 22,236 miles (35,786 kilometers) above Earth's equator, this position is a valuable spot for monitoring weather, communications and surveillance. Because the satellite orbits at the same speed that the Earth is turning, the satellite seems to stay in place over a single longitude, though it may drift north to south,” NASA wrote on its Earth Observatory website [3] .

* SSO (or SO): It is a Sun-synchronous orbit also called a heliosynchronous orbit is a nearly polar orbit around a planet, in which the satellite passes over any given point of the planet's surface at the same local mean solar time [4] .

* ES-L1 :At the Lagrange points the gravitational forces of the two large bodies cancel out in such a way that a small object placed in orbit there is in equilibrium relative to the center of mass of the large bodies. L1 is one such point between the sun and the earth [5] .

* HEO A highly elliptical orbit, is an elliptic orbit with high eccentricity, usually referring to one around Earth [6].

* ISS A modular space station (habitable artificial satellite) in low Earth orbit. It is a multinational collaborative project between five participating space agencies: NASA (United States), Roscosmos (Russia), JAXA (Japan), ESA (Europe), and CSA (Canada) [7]

* MEO Geocentric orbits ranging in altitude from 2,000 km (1,200 mi) to just below geosynchronous orbit at 35,786 kilometers (22,236 mi). Also known as an intermediate circular orbit. These are "most commonly at 20,200 kilometers (12,600 mi), or 20,650 kilometers (12,830 mi), with an orbital period of 12 hours [8]

* HEO Geocentric orbits above the altitude of geosynchronous orbit (35,786 km or 22,236 mi) [9]

* GEO It is a circular geosynchronous orbit 35,786 kilometres (22,236 miles) above Earth's equator and following the direction of Earth's rotation [10]

* PO It is one type of satellites in which a satellite passes above or nearly above both poles of the body being orbited (usually a planet such as the Earth [11]

some are shown in the following plot:

[![Orbits.png](https://i.postimg.cc/k5Vj93bg/Orbits.png)](https://postimg.cc/CZgCH273)

True Ocean means the mission outcome was successfully landed to a specific region of the ocean while False Ocean means the mission outcome was unsuccessfully landed to a specific region of the ocean. True RTLS means the mission outcome was successfully landed to a ground pad False RTLS means the mission outcome was unsuccessfully landed to a ground pad.True ASDS means the mission outcome was successfully landed to a drone ship False ASDS means the mission outcome was unsuccessfully landed to a drone ship. None ASDS and None None these represent a failure to land.

```
for i,outcome in enumerate(landing_outcomes.keys()):
    print(i,outcome)
    
#We create a set of outcomes where the second stage did not land successfully:
    bad_outcomes=set(landing_outcomes.keys()[[1,3,5,6,7]])
    
#Create a landing outcome label from Outcome column
landing_class=[0 if outcome in bad_outcomes else 1 for outcome in df['Outcome']]
df['Class']=landing_class
df[['Class']].head(8)
df.head(5)

#We can use the following line of code to determine the success rate:
df["Class"].mean()
df.to_csv("dataset_part_2.csv", index=False)
```
