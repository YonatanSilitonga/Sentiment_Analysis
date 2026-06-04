from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Setup Chrome
options = Options()
options.add_argument('--headless')
# options.add_argument('--disable-gpu')
# options.add_argument('--window-size=1920x1080')
driver = webdriver.Chrome(options=options)

# Buka halaman Maps
url = "https://www.google.com/maps/place/Se'i+Sapi+Lamalera/@-6.302937,106.6308827,14z/data=!4m10!1m2!2m1!1ssei+sapi!3m6!1s0x2e69fb98eac000bf:0x536e38354500d4c6!8m2!3d-6.302937!4d106.6689915!15sCghzZWkgc2FwaVoKIghzZWkgc2FwaZIBCnJlc3RhdXJhbnSqAT4QASoMIghzZWkgc2FwaSgmMh4QASIaQNVCGTaGyhZoLB-xLnboBXwNs2ENQGoKIfsyDBACIghzZWkgc2FwaeABAA!16s%2Fg%2F11qnrtt9s0?entry=ttu&g_ep=EgoyMDI1MDUxMS4wIKXMDSoASAFQAw%3D%3D"
driver.get(url)
time.sleep(5)

# Klik tombol More Reviews jika ada
try:
    more_reviews = WebDriverWait(driver, 10).until(
        EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Ulasan lainnya') or contains(text(), 'More reviews')]/ancestor::button"))
    )
    more_reviews.click()
    print("[INFO] Klik tombol More reviews.")
    time.sleep(3)
except:
    print("[INFO] Tidak ada tombol More reviews.")

# Loop scroll berdasarkan elemen terakhir
reviews_data = []
seen_reviews = set()
scroll_attempt = 0
max_scroll_attempts = 200
print("[INFO] Mulai scroll...")

while scroll_attempt < max_scroll_attempts and len(reviews_data) < 50:
    # Ambil semua container review
    review_containers = driver.find_elements(By.CSS_SELECTOR, 'div.jftiEf')

    for container in review_containers:
        try:
            username = container.find_element(By.CLASS_NAME, 'd4r55').text.strip()
        except:
            username = 'Unknown'

        try:
            review = container.find_element(By.CLASS_NAME, 'wiI7pd').text.strip()
        except:
            review = ''

        try:
            rating_element = container.find_element(By.CLASS_NAME, 'kvMYJc')
            rating = rating_element.get_attribute('aria-label').split()[0]  # contoh: "2 stars"
        except:
            rating = 'Unknown'

        # Cek duplikat review
        unique_key = (username, review)
        if review and unique_key not in seen_reviews:
            seen_reviews.add(unique_key)
            reviews_data.append({
                'user': username,
                'review': review,
                'rating': rating
            })

    # Scroll ke bawah dengan elemen terakhir
    if review_containers:
        driver.execute_script("arguments[0].scrollIntoView();", review_containers[-1])
    else:
        print("[INFO] Tidak ada container review ditemukan.")
        break

    print(f"[INFO] Total ulasan terkumpul: {len(reviews_data)}")
    time.sleep(2)
    scroll_attempt += 1

    if len(reviews_data) >= 50:
        print("[INFO] Sudah mencapai 50 ulasan, berhenti scrolling.")
        break

# Cetak hasil
print("\n=== Hasil Ulasan ===")
for i, item in enumerate(reviews_data[:50], 1):
    print(f"{i}. User: {item['user']}")
    print(f"   Rating: {item['rating']}")
    print(f"   Review: {item['review']}\n")

# Simpan ke CSV
df = pd.DataFrame(reviews_data)
df.to_excel('ulasan_sei_sapi_lamalera.xlsx', index=False)
print("[INFO] Data berhasil disimpan ke 'ulasan_sei_sapi_lamalera.xlsx'.")

driver.quit()
