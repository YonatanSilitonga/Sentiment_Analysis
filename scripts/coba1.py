from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

# Setting up Chrome options
chrome_options = Options()
# chrome_options.add_argument("--headless")  # Ensure GUI is off
# chrome_options.add_argument("--no-sandbox")
# chrome_options.add_argument("--disable-dev-shm-usage")

# Set path to chromedriver as per your configuration
webdriver_service = Service(ChromeDriverManager().install())

# Choose Chrome Browser
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# Open Google Maps
driver.get("https://www.google.com/maps")

# Find the search box using its HTML ID and enter the search term
search_box = driver.find_element(By.ID, "searchboxinput")
search_box.send_keys("restaurants in New York")
search_box.send_keys(Keys.ENTER)

# Wait for the results to load
import time
time.sleep(5)  # Adjust this delay based on your internet speed

# Function to scroll and load more results
def scroll_and_load(driver):
    results_list = driver.find_element(By.XPATH, '//*[@id="pane"]/div/div[1]/div/div/div[4]')
    for i in range(10):  # Adjust number of scrolls as needed
        driver.execute_script('arguments[0].scrollTop = arguments[0].scrollHeight', results_list)
        time.sleep(2)  # Adjust delay as needed

scroll_and_load(driver)

from bs4 import BeautifulSoup

# Get the page source after scrolling
page_source = driver.page_source

# Parse with BeautifulSoup
soup = BeautifulSoup(page_source, "html.parser")

# Find all the relevant data containers
data_containers = soup.find_all("div", class_="section-result-content")

# Extract details
results = []
for container in data_containers:
    name = container.find("h3", class_="section-result-title").text
    address = container.find("span", class_="section-result-location").text
    try:
        rating = container.find("span", class_="cards-rating-score").text
    except AttributeError:
        rating = "No rating"
    results.append({"name": name, "address": address, "rating": rating})