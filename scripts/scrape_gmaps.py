import os
import sys
import time
import argparse
import pandas as pd
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Default columns to match the project's dataset style
COLUMNS = ["nama", "tanggal", "ulasan", "rating", "label", "sumber"]

def setup_driver():
    """Menginisialisasi Chrome WebDriver dengan opsi terbaik untuk menghindari blokir."""
    options = webdriver.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    # Biarkan browser tetap terbuka agar user bisa melihat prosesnya
    
    # Gunakan WebDriver Manager untuk otomatis mendownload driver yang cocok
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def click_more_buttons(driver):
    """Menemukan dan mengklik tombol 'Lainnya' atau 'More' agar teks ulasan panjang tidak terpotong."""
    try:
        # Cari tombol dengan class w8nwRe yang biasanya merupakan tombol 'Lainnya'
        more_buttons = driver.find_elements(By.CSS_SELECTOR, "button.w8nwRe")
        clicked_count = 0
        for btn in more_buttons:
            if btn.is_displayed():
                try:
                    driver.execute_script("arguments[0].click();", btn)
                    clicked_count += 1
                except Exception:
                    pass
        if clicked_count > 0:
            print(f"  [+] Mengklik {clicked_count} tombol 'Lainnya' untuk membaca ulasan lengkap.")
    except Exception:
        pass

def scrape_gmaps_reviews(url, max_reviews=300, output_name=None):
    print("=" * 60)
    print("MEMBERSIHKAN & MENYIAPKAN BROWSER SELENIUM...")
    print("=" * 60)
    
    driver = setup_driver()
    driver.get(url)
    
    print("\n[!] BROWSER TELAH DIBUKA.")
    
    # 1. Klik tombol 'Ulasan lainnya' / 'More reviews' jika ada (seperti di coba.py)
    try:
        more_reviews = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//span[contains(text(), 'Ulasan lainnya') or contains(text(), 'More reviews') or contains(text(), 'ulasan') or contains(text(), 'reviews')]/ancestor::button"))
        )
        more_reviews.click()
        print("[+] Mengklik tombol 'Ulasan lainnya' / 'More reviews'...")
        time.sleep(3)
    except Exception:
        print("[INFO] Tidak mendeteksi atau tidak memerlukan tombol 'More reviews' untuk diklik.")
    
    print("[!] Menunggu elemen ulasan pertama dimuat (maksimal 20 detik)...")
    
    # Tunggu sampai salah satu ulasan dengan class jftiEf muncul
    try:
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".jftiEf"))
        )
        print("[+] Ulasan terdeteksi! Memulai proses scrolling...")
    except Exception:
        print("[-] Peringatan: Tidak dapat mendeteksi ulasan. Pastikan Anda berada di halaman ulasan.")
        print("[!] Menunggu manual... Silakan klik tab ulasan di jendela browser yang terbuka.")
        time.sleep(10)

    # Temukan panel scrollable ulasan (sebagai fallback)
    scrollable_div = None
    try:
        scrollable_div = driver.find_element(By.XPATH, '//div[@role="feed"]')
    except Exception:
        try:
            scrollable_div = driver.find_element(By.CSS_SELECTOR, ".m6QErb.DxyBCb.kA9KIf.dS8AEf.XiKgde")
        except Exception:
            pass

    if not scrollable_div:
        try:
            scrollable_div = driver.find_element(By.TAG_NAME, "body")
        except Exception:
            pass

    # Proses Scroll untuk me-load ulasan
    reviews_count = 0
    no_change_count = 0
    scroll_attempt = 0
    max_scroll_attempts = 300
    
    while reviews_count < max_reviews and scroll_attempt < max_scroll_attempts:
        # Coba scroll dengan mencari container scrollable secara dinamis & scroll ke bawah
        try:
            driver.execute_script("""
                var review = document.querySelector('.jftiEf');
                if (review) {
                    var parent = review.parentElement;
                    while (parent && parent.tagName !== 'BODY') {
                        var style = window.getComputedStyle(parent);
                        var overflow = style.getPropertyValue('overflow') + style.getPropertyValue('overflow-y');
                        if (overflow.indexOf('auto') !== -1 || overflow.indexOf('scroll') !== -1) {
                            parent.scrollTop = parent.scrollHeight;
                            break;
                        }
                        parent = parent.parentElement;
                    }
                }
            """)
        except Exception:
            pass

        # Tambahan: scrollIntoView pada ulasan terakhir agar lebih meyakinkan
        current_review_elements = driver.find_elements(By.CSS_SELECTOR, ".jftiEf")
        if current_review_elements:
            try:
                driver.execute_script("arguments[0].scrollIntoView();", current_review_elements[-1])
            except Exception:
                if scrollable_div:
                    try:
                        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
                    except Exception:
                        pass
        else:
            if scrollable_div:
                try:
                    driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", scrollable_div)
                except Exception:
                    pass

        time.sleep(2.5) # Jeda agar ulasan baru ter-load
        
        # Klik tombol 'Lainnya' jika ada
        click_more_buttons(driver)
        
        # Hitung jumlah ulasan terkumpul saat ini
        new_review_elements = driver.find_elements(By.CSS_SELECTOR, ".jftiEf")
        new_count = len(new_review_elements)
        print(f"\r-> Loaded {new_count}/{max_reviews} ulasan...", end="")
        
        if new_count == reviews_count:
            no_change_count += 1
            if no_change_count >= 6: # Jika 6 kali scroll tidak ada ulasan baru, berarti ulasan habis
                print("\n[+] Semua ulasan di Google Maps untuk tempat ini telah dimuat.")
                break
        else:
            no_change_count = 0
            reviews_count = new_count
            
        scroll_attempt += 1

    print(f"\n[+] Mulai mengekstrak data dari {reviews_count} ulasan...")
    
    review_elements = driver.find_elements(By.CSS_SELECTOR, ".jftiEf")
    data_list = []
    
    for idx, element in enumerate(review_elements[:max_reviews]):
        try:
            # 1. Nama Reviewer
            nama = "Anonymous"
            try:
                # Class nama reviewer biasanya .d4r55 atau .XpcR2c atau tombol dengan class tertentu
                name_elem = element.find_element(By.CSS_SELECTOR, ".d4r55")
                nama = name_elem.text.strip()
            except Exception:
                try:
                    name_elem = element.find_element(By.CSS_SELECTOR, ".XpcR2c")
                    nama = name_elem.text.strip()
                except Exception:
                    pass
            
            # 2. Tanggal/Waktu
            tanggal = ""
            try:
                date_elem = element.find_element(By.CSS_SELECTOR, ".rsqaWe")
                tanggal = date_elem.text.strip()
            except Exception:
                pass
            
            # 3. Teks Ulasan
            ulasan = ""
            try:
                review_text_elem = element.find_element(By.CSS_SELECTOR, ".wiI7pd")
                ulasan = review_text_elem.text.strip()
            except Exception:
                pass
            
            # 4. Rating Stars (Mencoba class kvMYJc seperti coba.py dan kv7c6e)
            rating = None
            for rating_class in [".kvMYJc", ".kv7c6e"]:
                try:
                    rating_elem = element.find_element(By.CSS_SELECTOR, rating_class)
                    aria_label = rating_elem.get_attribute("aria-label")
                    if aria_label:
                        # Ambil angka dari teks "5 stars" atau "5 bintang"
                        rating = int(aria_label.split()[0])
                        break
                except Exception:
                    pass
            
            if rating is None:
                try:
                    # Alternatif cari dengan selector bintang umum
                    rating_elem = element.find_element(By.CSS_SELECTOR, "span[aria-label*='bintang'], span[aria-label*='star']")
                    aria_label = rating_elem.get_attribute("aria-label")
                    rating = int(aria_label.split()[0])
                except Exception:
                    pass
            
            # Jika ulasannya kosong, tidak perlu dimasukkan ke dataset sentimen
            if not ulasan:
                continue
                
            data_list.append({
                "nama": nama,
                "tanggal": tanggal,
                "ulasan": ulasan,
                "rating": rating,
                "label": "", # Kosongkan dahulu untuk manual pelabelan atau auto-prediksi
                "sumber": "google-maps-scraped"
            })
            
        except Exception as e:
            print(f"\n[-] Gagal mengekstrak ulasan ke-{idx+1}: {e}")
            
    driver.quit()
    print("[+] Browser ditutup.")
    
    if not data_list:
        print("[-] Tidak ada ulasan teks yang berhasil diekstrak.")
        return
        
    df = pd.DataFrame(data_list)
    # Prediksi sentimen dinonaktifkan sesuai permintaan user, kolom 'label' akan tetap kosong
    print("\n[!] Pengingat: Kolom 'label' dibiarkan kosong untuk pelabelan manual.")

    # Simpan hasil ke Excel
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not output_name:
        output_name = f"gmaps_scraped_{date_str}.xlsx"
    elif not output_name.endswith(".xlsx"):
        output_name = f"{output_name}.xlsx"
        
    output_dir = os.path.join("Merged_Excel")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_name)
    
    # Urutkan kolom agar pas
    df = df[COLUMNS]
    df.to_excel(output_path, index=False)
    
    print("=" * 60)
    print("PROSES SCRAPING SELESAI")
    print(f"Jumlah ulasan berhasil disimpan : {len(df)}")
    print(f"File disimpan di                : {os.path.abspath(output_path)}")
    print("=" * 60)
    print("\nTips:")
    print(f"1. Anda dapat menggabungkannya ke dataset master dengan menjalankan:")
    print("   python experiments/merge_xlsx_merged_excel.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Maps Reviews Scraper using Selenium")
    parser.add_argument("--url", required=True, help="URL Google Maps ulasan tempat wisata")
    parser.add_argument("--max", type=int, default=50, help="Maksimum ulasan yang ingin diambil (default: 50)")
    parser.add_argument("--output", default=None, help="Nama file hasil excel (contoh: ulasan_sipisopiso.xlsx)")
    
    args = parser.parse_args()
    scrape_gmaps_reviews(args.url, args.max, args.output)
