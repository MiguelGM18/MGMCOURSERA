### MIGUEL GONZALEZ
# import requests

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
df.to_csv('spacex_web_scraped.csv', index=False)
```

## Contributing


## License

