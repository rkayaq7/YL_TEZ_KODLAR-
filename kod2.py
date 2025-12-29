import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import re

# NLTK paketlerini sessizce indir
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

stop_words = set(stopwords.words('turkish'))

data2 = {
    "Katılımcı": ["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10"],
    "Yanıt": [
        "Bence günlük hayatımızı çok kolaylaştırıyor. Sabah uyandığımda alarm uygulamam uyku analizime göre çalıyor. Ulaşımda Google Maps trafiği tahmin ediyor ve hızlı rota sunuyor. Ders çalışırken de ChatGPT çok işe yarıyor; anlamadığım çevre ve ekoloji konularını soruyorum, farklı örneklerle açıklıyor. Sosyal medyada ve alışveriş sitelerinde öneri sistemleri sayesinde ilgimi çeken içerikleri hızlıca bulabiliyorum. Kısaca, hayatımı daha düzenli ve verimli hale getiriyor.",
        "Şahsen çok farkında değilim. Telefonun bazı şeyleri otomatik yapması var, mesela mesaj öneriyor veya müzik listesi hazırlıyor. Onu yapay zekâya bağlıyorum. Ama tam emin değilim, bazen garip oluyor, bazen işe yarıyor. Ödevlerde de bazen ChatGPT kullanıyorum ama bazen saçma cevaplar veriyor.",
        "Günlük hayatımda etkisini fark etmek kolay. Sabah kalkıp telefonuma baktığımda, alarm uygulaması uykumu analiz edip en uygun saatte çalıyor. Yolculuk yaparken Google Maps veya Yandex'in trafik tahminleri yapması da yapay zekânın işi. Sosyal medya ve e-ticaret sitelerinde karşıma çıkan öneriler de kişisel alışkanlıklarımı analiz edip geliyor. Ders çalışırken de fark ediyorum. Özellikle anlamadığım konuları ChatGPT'ye soruyorum.",
        "Eskiden fark etmezdim ama artık gün içinde defalarca yapay zekâ ile etkileşime girdiğimi fark ediyorum. Spotify tam tarzıma uygun şarkılar öneriyor. E-ticaret siteleri ne almak isteyeceğimi tahmin ediyor. Sosyal medyada ilgilendiğim konuları öğrenip karşıma onlarca içerik çıkarıyor. Ders çalışırken ChatGPT'ye soruyorum.",
        "Açıkçası ben fark ettikçe şaşırıyorum. Sabah telefonumu açtığım andan itibaren yapay zeka bir şekilde devreye giriyor. Alarm uygulamam bile benim uyku düzenimi öğrenmiş gibi bazen yumuşak bir müzikle başlatıyor. Sonra Google Maps trafik durumunu tahmin ediyor. Ders çalışırken anlamadığım konuyu ChatGPT'ye soruyorum.",
        "Günlük yaşamımda farkında olmadan çok kullanıyorum. Alışveriş siteleri ilgi alanıma göre ürün öneriyor. Yemek tarifleri uygulamaları, hangi malzemeleri kullanabileceğimi tahmin edip öneriler sunuyor. Laboratuvarlarda bazı analiz cihazları da yapay zekâ kullanıyor.",
        "Gerçekten çok etkisi var. Sabah kalktığımda alarm uygulaması uyku düzenime göre ayarlanabiliyor. Yol tarifi için Google Maps veya Yandex'in trafiği tahmin etmesi de yapay zekânın işi. Sosyal medyada ve alışveriş sitelerinde gördüğüm öneriler, beğendiğim içeriklere göre geliyor. Ders çalışırken anlamadığım konuyu ChatGPT'ye sorabiliyorum.",
        "Bence çok büyük bir etkisi var. Telefonumun uyku analizi yapıp alarmı ona göre çalması bile yapay zekânın etkisi. Ders çalışırken ChatGPT'den tekrar ettiriyorum. Sosyal medya ve e-ticaret önerileri günlük alışkanlıklarımı şekillendiriyor.",
        "Bence baya etkisi var. Telefonum, bilgisayarım ve sosyal medya gibi şeyler yapay zekâ sayesinde daha kişisel oluyor. Spotify müzik listemi kendisi oluşturuyor, YouTube ilgi alanıma göre video çıkarıyor. Ödevlerde de bazen ChatGPT kullanıyorum.",
        "Günlük hayatımda fark etmesem bile sürekli karşıma çıkıyor. Telefonumda yazdığım şeyin otomatik tamamlanması, YouTube'un öneriler sunması, navigasyonun en kısa yolu bulması yapay zekayla oluyor. Ders çalışırken ChatGPT'den ödev açıklaması isteyebiliyorum."]
}

df2 = pd.DataFrame(data2)
df2.head()

# NLTK hatası durumunda çalışacak alternatif temizleme fonksiyonu
def temizle(metin):
    if not isinstance(metin, str) or metin == "":
        return []
    
    try:
        metin = metin.lower()
        metin = ''.join([c for c in metin if c not in string.punctuation])
        kelimeler = word_tokenize(metin)
        kelimeler = [k for k in kelimeler if k not in stop_words]
        return kelimeler
    except:
        # NLTK hatası durumunda basit yöntem
        metin = metin.lower()
        metin = re.sub(r'[^\w\s]', '', metin)
        kelimeler = metin.split()
        kelimeler = [k for k in kelimeler if k not in stop_words]
        return kelimeler

df2['Token'] = df2['Yanıt'].apply(temizle)

# Temaları basit kurallarla belirleme
temalar = []
for tokens in df2['Token']:
    tema_listesi = []
    if any(k in tokens for k in ["telefon", "youtube", "spotify", "sosyal", "alışveriş", "maps", "google", "yandex", "uyku", "alarm", "navigasyon"]):
        tema_listesi.append("Günlük yaşamda kullanım")
    if any(k in tokens for k in ["ders", "chatgpt", "ödev", "proje", "laboratuvar", "anlamadığım", "konu", "eğitim", "çalışırken"]):
        tema_listesi.append("Eğitim ve öğrenme")
    if any(k in tokens for k in ["kolay", "verimli", "hızlı", "zaman", "düzenli", "tasarruf", "kişisel", "otomatik"]):
        tema_listesi.append("Zaman tasarrufu ve verimlilik")
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df2['Temalar'] = temalar

print("Soru 2 Tematik Analiz Tablosu:")
print(df2[['Katılımcı','Yanıt','Temalar']])

# Tema dağılımını göster
print("\nTema Dağılımı:")
from collections import Counter
tema_frekans = Counter([tema for sublist in [t.split(', ') for t in df2['Temalar']] for tema in sublist])
for tema, sayi in tema_frekans.items():
    print(f"{tema}: {sayi}")