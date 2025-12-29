import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# NLTK verilerini indir
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

stop_words = set(stopwords.words('turkish'))

# Katılımcı verisi
data = {
    "Katılımcı": ["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10"],
    "Yanıt": [
        "Genelde YouTube ve Coursera'dan temel ve orta seviyede yapay zekâ içeriklerini takip ediyorum. Özellikle çevre uygulamalarına odaklanan videoları izliyorum. Üniversitede bazı yazılımların su kirliliği tahmininde yapay zekâ kullandığını gördük. ChatGPT'yi de ödev ve araştırmalarda kavramları anlamak için kullanıyorum.",
        "Ara sıra YouTube izliyorum; arkadaşlardan duydukları; ChatGPT'ye arada bakıyorum.",
        "YouTube videoları ve seminerler; ChatGPT ile pratik öğrenme; temel seviyede bilgi.",
        "Basit anlatımlı YouTube videoları; Endüstri Mühendisliği Kulübü etkinliği; ChatGPT.",
        "YouTube videoları; kulüp etkinlikleri; ChatGPT ile örneklerle öğrenme.",
        "YouTube, Coursera ve çevrimiçi makaleler; derslerde YZ örnekleri; simülasyon programları.",
        "Temel seviyede YouTube videoları; seminerler; ChatGPT.",
        "YouTube, Coursera, Kaggle; ChatGPT.",
        "YouTube ve Udemy; üniversite dersleri (Robotik Sistemler, Üretim Optimizasyonu); ChatGPT.",
        "YouTube; arkadaşlar ve ChatGPT ile deneme; resmi eğitim yok." ]
}

df = pd.DataFrame(data)

# Temizleme ve tokenizasyon fonksiyonu
def temizle(metin):
    if not isinstance(metin, str) or metin == "":
        return []
    
    try:
        metin = metin.lower()
        metin = ''.join([c for c in metin if c not in string.punctuation])
        kelimeler = word_tokenize(metin)
        kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
        return kelimeler
    except:
        # NLTK hatası durumunda basit yöntem
        metin = metin.lower()
        metin = re.sub(r'[^\w\s]', '', metin)
        kelimeler = metin.split()
        kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
        return kelimeler

# HATA DÜZELTİLDİ: apply yerine apply(temizle) kullanıldı
df['Token'] = df['Yanıt'].apply(temizle)

# Temaları basit kurallarla çıkarma
temalar = []
for tokens in df['Token']:
    tema_listesi = []
    if any(kelime in tokens for kelime in ["youtube", "coursera", "udemy", "makale", "kaggle"]):
        tema_listesi.append("Çevrimiçi öğrenme platformları")
    if any(kelime in tokens for kelime in ["seminer", "kulüp", "etkinlik", "ders", "üniversite"]):
        tema_listesi.append("Seminer ve etkinlikler")
    if any(kelime in tokens for kelime in ["chatgpt", "araç", "pratik"]):
        tema_listesi.append("YZ öğrenme araçları")
    if any(kelime in tokens for kelime in ["simülasyon", "proje", "uygulama", "yazılım", "programları"]):
        tema_listesi.append("Akademik/proje uygulamaları")
    if any(kelime in tokens for kelime in ["deneme", "arkadaş", "arada", "temel", "basit"]):
        tema_listesi.append("Sınırlı kullanım/deneme")
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df['Temalar'] = temalar

# Tabloyu göster
print("\nSoru 3 Tematik Analiz Tablosu:\n")
print(df[['Katılımcı', 'Yanıt', 'Temalar']])

# Tema dağılımını göster
print("\nTema Dağılımı:")
tema_frekans = Counter([tema for sublist in [t.split(', ') for t in df['Temalar']] for tema in sublist])
for tema, sayi in tema_frekans.items():
    print(f"{tema}: {sayi}")