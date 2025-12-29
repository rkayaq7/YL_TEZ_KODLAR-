import pandas as pd
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re

# Eksik NLTK paketlerini indir
try:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')  # Bu yeni paketi ekleyin
except:
    pass

stop_words = set(stopwords.words('turkish'))

# Katılımcı verisi
data = {
    "Katılımcı": ["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10"],
    "Yanıt": [
        "Bence yapay zekâ, bilgisayarların veya sistemlerin çok büyük veriler üzerinden analiz yapıp, kendi mantığıyla sonuç üretebilmesi demek. İnsan gibi düşünmüyor ama bazı kararları bağımsız alabiliyor. Çevre mühendisliği açısından baktığımda, örneğin su kalitesi ölçümlerini analiz eden veya hava kirliliği tahminleri yapan yazılımlar yapay zekâ kullanıyor. Günlük hayatta ise telefonumda fotoğrafları otomatik etiketlemesi veya YouTube'un öneri sistemleri gibi şeylerde yapay zekâ var. Yani hem akademik hem günlük kullanımda hayatımı etkiliyor.",
        "Bence yapay zekâ, bilgisayarların kendi kafasına göre bir şeyler yapabilmesi… ya da mantıklı gibi görünen şeyler söylemesi gibi bir şey. Tam olarak anlamadım aslında, ama bazı makineler kendi kendine karar veriyormuş gibi çalışıyor gibi geliyor. Mesela telefonum bazen yanlışlıkla fotoğrafları grupluyor, sanırım bu da yapay zekâ falan.",
        "Bence yapay zekâ, bilgisayarların ve bazı makinelerin kendi başlarına karar verebilmesi veya öğrenebilmesi demek. Çok teknik bilmiyorum ama örneklerle daha iyi anlıyorum. Mesela telefonum fotoğrafları otomatik grupluyor, sosyal medya ilgi alanımı tahmin edip içerik gösteriyor… bunlar yapay zekâ. Bir de inşaat alanında, bazı projelerde planlama ve analiz için bilgisayarların öneri sunması gibi bir şekilde de kullanılıyor.",
        "Bence yapay zekâ, bilgisayarların veya makinelerin insan gibi düşünebilmeye yakın davranması demek. Tam olarak nasıl çalıştığını bilmiyorum ama öğrendikçe fark ediyorum ki, yapay zekâ bir şeyi çok fazla veri üzerinden öğrenip, tahmin ve öneri yapabiliyor. Mesela telefonumun fotoğrafları otomatik etiketlemesi, sosyal medyanın ilgi alanıma göre içerik önermesi veya Spotify'ın müzik listesi hazırlaması gibi şeyler yapay zekâ sayesinde oluyor. Benim alanımla ilgili olarak ise, elektrik devreleri veya robotik sistemlerde bir şeyin kendini optimize etmesi, sensörlerden gelen veriyi analiz etmesi gibi durumlar da yapay zekâya örnek. Kısacası benim için, günlük hayatı kolaylaştıran ve mühendislik alanında işleri hızlandıran bir teknoloji.",
        "Yapay zeka benim için, bilgisayarların sadece tek bir işi yapan makine olmaktan çıkıp biraz daha insan mantığına yaklaşması gibi bir şey. Çok fazla veriyi inceleyip bir sonuç çıkarabilen, tahmin yapabilen ve hatta bazen öneriler sunabilen bir sistem.",
        "Yapay zekâ dediğimiz şey benim gözümde, bilgisayarların sadece verilen komutları çalıştırması değil de, biraz daha düşünüyormuş gibi davranması aslında. Çok fazla veri görüp, onlardan bir şey öğrenip sonra bir sonuca varması gibi düşünülüyor. Mesela telefonumun galerisinde aynı tip fotoğrafları kategorilere ayırması veya alışveriş sitelerinin öneriler sunması…",
        "Bence yapay zekâ, bilgisayarların sadece verilen komutları değil, verilerden öğrendikleri şeyleri kullanarak kendi kendine bazı kararları verebilmesi demek. İnsan beyninin çalışma mantığını taklit etmeye çalışan bir teknoloji türü.",
        "Ben yapay zekâyı, bilgisayarların veya makinelerin sadece programlandığı şekilde değil, biraz kendi mantığını kullanarak karar verebilmesi şeklinde düşünüyorum. Çok veri üzerinden analiz yapıp, örüntüleri fark edebiliyor ve önerilerde bulunabiliyor. Üniversitedeki üretim derslerinde bazı üretim hatlarında malzeme akışını ve üretim planlamasını yapay zekâ optimizasyon algoritmaları yönetiyor. Günlük hayatta ise telefonumun fotoğraf uygulamasının yüzleri tanıyıp albümlere ayırması ya da online alışverişte ilgimi çekebilecek ürünleri önermesi gibi örneklerle de karşılaşıyorum.",
        "Şöyle söyleyebilirim, yapay zekâ bence bilgisayarların insan gibi düşünebilmesi… ya da karar verebilmesi gibi bir şey. Bazı makineler insan yerine düşünüp karar veriyormuş gibi çalışıyor. Telefonumda bazı şeyleri otomatik olarak yapıyor, öneriler çıkarıyor.",
        "Yapay zekâ benim için makinelerin insan gibi düşünme yeteneği kazanması demek. Örneğin sesli asistanlar, otonom araçlar gibi teknolojiler yapay zekâ sayesinde çalışıyor."
    ]
}

df = pd.DataFrame(data)

# Alternatif temizleme fonksiyonu - NLTK olmadan
def temizle(metin):
    if not isinstance(metin, str) or metin == "":
        return []
    
    metin = metin.lower()
    # Noktalama işaretlerini kaldır
    metin = re.sub(r'[^\w\s]', '', metin)
    # Basit boşluk split ile tokenize et
    kelimeler = metin.split()
    # Stop words'leri kaldır
    kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
    return kelimeler

# VEYA NLTK ile çalışan güncellenmiş fonksiyon
def temizle_nltk(metin):
    if not isinstance(metin, str) or metin == "":
        return []
    
    try:
        metin = metin.lower()
        metin = ''.join([c for c in metin if c not in string.punctuation])
        kelimeler = word_tokenize(metin)
        kelimeler = [kelime for kelime in kelimeler if kelime not in stop_words]
        return kelimeler
    except:
        # NLTK hatası durumunda basit yönteme dön
        return temizle(metin)

# Basit yöntemi kullan
df['Token'] = df['Yanıt'].apply(temizle)

# Temaları basit kurallarla çıkarma
temalar = []
for tokens in df['Token']:
    tema_listesi = []
    if any(kelime in tokens for kelime in ["günlük", "telefon", "hayat", "öneri", "fotoğraf", "sosyal", "youtube", "spotify"]):
        tema_listesi.append("Günlük yaşam ve YZ")
    if any(kelime in tokens for kelime in ["veri", "analiz", "hesaplama", "optimizasyon", "algoritma", "mühendislik", "akademik", "teknoloji"]):
        tema_listesi.append("Teknik/akademik kullanım")
    if any(kelime in tokens for kelime in ["insan", "düşün", "karar", "öğren", "mantık", "beyin", "taklit"]):
        tema_listesi.append("YZ işleyiş algısı")
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df['Temalar'] = temalar

# Tablo oluşturma
print("\nSoru 1 Tematik Analiz Tablosu:\n")
print(df[['Katılımcı', 'Yanıt', 'Temalar']])

# Ek olarak temaların frekansını göster
print("\nTema Dağılımı:")
tema_frekans = Counter([tema for sublist in [t.split(', ') for t in df['Temalar']] for tema in sublist])
for tema, sayi in tema_frekans.items():
    print(f"{tema}: {sayi}")