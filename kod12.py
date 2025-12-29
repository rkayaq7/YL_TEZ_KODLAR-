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

# Katılımcı ve yanıt verileri
data2 = {
    "Katılımcı": [
        "Çevre Mühendisliği (K)", "Çevre Mühendisliği (E)", "Elekt.-Elektr. Mühendisliği (E)",
        "Endüstri Mühendisliği (K)", "Gıda Mühendisliği (E)", "Gıda Mühendisliği (K)",
        "İnşaat Mühendisliği (K)", "İnşaat Mühendisliği (E)", "Makine Mühendisliği (K)",
        "Makine Mühendisliği (E)"
    ],
    "Yanıt": [
        "Yeni roller: Çevresel veri analizi ve optimizasyon mühendisi, Akıllı şehir ve su kaynakları yönetimi uzmanı, Hava kirliliği ve enerji verimliliği danışmanı, Simülasyon ve tahmin modelleri geliştiren mühendis. Teknolojiye hakim mühendisler daha öne çıkacak.",
        "Belki sensörleri veya makineleri kontrol eden işler olabilir. Robotik falan olursa insanlar bunları takip eder. Ama tam emin değilim.",
        "Yapay zekâ destekli robotik sistem tasarımı ve optimizasyonu, Akıllı enerji sistemleri uzmanlığı, Otomasyon ve kontrol sistemleri analisti, Veri analizi ve tahmin modelleri geliştiren mühendisler.",
        "Yapay zeka destekli planlama uzmanı, Veri analitiği mühendisi, Yapay zekâ tabanlı kalite kontrol analisti, Dijital dönüşüm uzmanı, Akıllı fabrikalar için süreç tasarımı mühendisleri.",
        "Yapay zekâ destekli üretim hatlarını yöneten mühendisler, Verileri analiz edip üretimi optimize eden uzmanlar, Kalite kontrol sistem operatörleri, Gıda güvenliği risk tahmini yapan analistler.",
        "Akıllı üretim hattı kalite kontrol mühendisi, Veri analizi ve tahmin modelleri geliştiren gıda mühendisi, Laboratuvar analiz ve otomasyon uzmanı, Gıda üretim süreçlerini optimize eden danışman.",
        "Yapay zekâ destekli proje planlama uzmanı, Dijital şantiye yönetimi mühendisi, Yapay zekâ destekli kalite kontrol analisti, Veri analitiği ve maliyet optimizasyonu uzmanı.",
        "Yapı Sağlığı İzleme Uzmanı, Dijital Şantiye Analisti, AI Tabanlı Proje Planlama Mühendisi.",
        "Robotik sistem tasarım ve optimizasyon mühendisi, Akıllı üretim hattı planlama uzmanı, Enerji verimliliği ve simülasyon analisti, Veri analizi ve üretim performansını artıran mühendis.",
        "Belki robotik sistemleri yöneten işler olabilir. Fabrikada makineleri optimize eden kişiler falan." ]
}

df2 = pd.DataFrame(data2)

# Tokenization ve stop-word temizleme
stop_words = set(stopwords.words('turkish'))

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

# Geliştirilmiş tematik kodlama
temalar = []
for tokens in df2['Token']:
    tema_listesi = []
    
    if any(k in tokens for k in ["robotik", "otomasyon", "sistem", "sensör", "kontrol", "tasarım"]):
        tema_listesi.append("Robotik ve otomasyon")
    
    if any(k in tokens for k in ["veri", "tahmin", "analiz", "analitik", "model", "simülasyon", "tahmin"]):
        tema_listesi.append("Veri analizi ve tahmin modelleri")
    
    if any(k in tokens for k in ["planlama", "proje", "dijital", "optimizasyon", "süreç", "yönetim", "şantiye", "fabrika"]):
        tema_listesi.append("Proje ve süreç yönetimi")
    
    if any(k in tokens for k in ["kalite", "laboratuvar", "güvenlik", "kontrol", "operatör", "performans"]):
        tema_listesi.append("Kalite kontrol ve üretim optimizasyonu")
    
    if any(k in tokens for k in ["enerji", "verimlilik", "çevre", "çevresel", "hava", "su", "kaynak"]):
        tema_listesi.append("Enerji verimliliği ve çevre uygulamaları")
    
    if any(k in tokens for k in ["belki", "emin", "değilim", "olabilir", "falan"]):
        tema_listesi.append("Belirsizlik veya spekülasyon")
    
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df2['Temalar'] = temalar

print("Soru 10: Gelecekteki Mühendislik Rolleri - Tematik Analiz")
print("=" * 90)

# Sonuçları göster
result_df = df2[['Katılımcı','Yanıt','Temalar']]
print(result_df.to_string(index=False))

# Tema frekanslarını hesapla
from collections import Counter
print("\n" + "=" * 90)
print("TEMA FREKANSLARI")
print("=" * 90)

tema_frekans = Counter()
for tema in df2['Temalar']:
    temas = [t.strip() for t in tema.split(",")]
    for t in temas:
        if t != "Tema bulunamadı":
            tema_frekans[t] += 1

for tema, sayi in tema_frekans.most_common():
    print(f"{tema}: {sayi} katılımcı")

# Bölümlere göre tema dağılımı
print("\n" + "=" * 90)
print("KATILIMCI BAZINDA DETAYLI ANALİZ")
print("=" * 90)

for _, row in df2.iterrows():
    print(f"\n{row['Katılımcı']}:")
    print(f"Yanıt: {row['Yanıt']}")
    print(f"Temalar: {row['Temalar']}")
    print("-" * 80)

# Özet istatistikler
print("\n" + "=" * 90)
print("ÖZET İSTATİSTİKLER")
print("=" * 90)
print(f"Toplam katılımcı sayısı: {len(df2)}")
print(f"En yaygın gelecek rolü: {tema_frekans.most_common(1)[0][0] if tema_frekans else 'Yok'}")
print(f"En yaygın rol katılımcı sayısı: {tema_frekans.most_common(1)[0][1] if tema_frekans else 0}")

# Bölüm bazında temalar
print("\n" + "=" * 90)
print("BÖLÜMLERE GÖRE TEMALAR")
print("=" * 90)
for _, row in df2.iterrows():
    bolum = row['Katılımcı'].split(' ')[0]  # Bölüm adını al
    print(f"{bolum}: {row['Temalar']}")