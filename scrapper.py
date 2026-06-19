from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup as bs
from urllib.parse import urljoin
import time
import random
import pandas as pd

url = "https://www.beeradvocate.com/beer/styles/116/"
base_url = "https://www.beeradvocate.com"

links_to_download = []

beers = 4950
with sync_playwright() as p:
    browser = p.chromium.launch(headless = False,args=["--disable-blink-features=AutomationControlled"])
    page = browser.new_page()
    for start_index in range(0,beers, 50):
        page.goto(f"{url}?sort=revsD&start={start_index}")

        html = page.content()
        soup = bs(html, 'html.parser')

        all_rows = soup.find_all('tr')
        beer_rows = all_rows[3:-1]

        
        for row in beer_rows:
            first_column = row.find('td')
            if first_column:
                first_link = first_column.find('a')
                if first_link and 'href' in first_link.attrs:
                    relative_link = first_link['href']
                    
                    beer_name = first_link.text.strip()
                    full_link = urljoin(base_url, relative_link)
                    links_to_download.append({'Name': beer_name, 'url': full_link})
    beers_data = []

    for link in links_to_download:
        page.goto(link['url'])
        beer_html = page.content()
        beer_soup = bs(beer_html, features = 'html.parser')
        beer_stats = {'Name': link['Name']}

        dl = beer_soup.find('dl', class_ = 'beerstats')

        if dl:
            all_dts = beer_soup.find_all('dt')
            for dt in all_dts:
                key = dt.text.strip().replace(':', '')

                if not key: 
                    key = 'Region'

                if key not in ['From','Region', 'Style', 'ABV', 'Score', 'Avg', 'Ratings', 'Status', "Rated", 'Added', 'Wants', 'Gots']:
                    continue

                dd = dt.find_next_sibling('dd')
                if dd:
                    value = dd.text.strip()
                    beer_stats[key] = value
        beers_data.append(beer_stats)
    
    beers_data_frame = pd.DataFrame(beers_data)
    print(beers_data_frame.head(4))    
    file_path = 'beers_data.csv'
    beers_data_frame.to_csv(file_path, index = False)
