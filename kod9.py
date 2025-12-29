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

# Katılımcı verileri
data = {
    "Katılımcı": [
        "Çevre3", "Çevre1", "EE4", "Endüstri3", "Gıda2", "Gıda3", 
        "İnşaat3K", "İnşaat3E", "Makine3K", "Makine1E"
    ],
    "Yanıt": [
        "Hava ve su kalitesi tahmini, atık yönetimi ve enerji verimliliği uygulamaları ilgimi çekiyor. Günlük hayatta ise ChatGPT gibi sohbet botları ve öneri sistemleri çok işime yarıyor; derslerde, projelerde ve araştırmalarda pratik olarak kullanıyorum.",
        "Bilmiyorum açıkçası, fazla ilgim yok. Ama ChatGPT gibi sohbet botları veya oyunlardaki yapay zekâ karakterleri ilgimi çekiyor. Bir de YouTube öneri sistemi eğlenceli bazen.",
        "En çok robotik ve otomasyon sistemlerine ilgi duyuyorum. Günlük hayatta sohbet botları ve öneri sistemleri de ilgimi çekiyor. Ders ve projelerde çok işe yarıyorlar.",
        "En çok tahmin modellerine ve öneri sistemlerine ilgi duyuyorum. Ayrıca optimizasyon modelleri de ilgimi çekiyor. Bir de sohbet botlarını seviyorum.",
        "En çok görsel tanıma yani görüntü işleme ilgimi çekiyor. Bir de sohbet botları ilgimi çekiyor, derslerde çok işime yarıyor.",
        "Kalite kontrol ve üretim optimizasyonu ilgimi çekiyor. Ayrıca laboratuvar analizleri ve veri tahmin modelleri de çok ilginç geliyor. ChatGPT gibi sohbet botları ve yemek tarifleri öneren uygulamalar ilgimi çekiyor.",
        "En çok inşaat ve planlama alanında kullanılan tahmin ve optimizasyon modellerine ilgi duyuyorum. Ayrıca görsel analiz uygulamaları da ilgimi çekiyor. Günlük hayatta sohbet botları ve öneri sistemleri ilgimi çekiyor.",
        "En çok görüntü işleme ilgimi çekiyor. Ayrıca makine öğrenmesini de öğrenmek istiyorum.",
        "Robotik sistemler, üretim hattı optimizasyonu ve enerji verimliliği uygulamaları ilgimi çekiyor. Ayrıca simülasyon ve tahmin modelleri ile çalışmak da ilgimi çekiyor. ChatGPT gibi sohbet botları ve öneri sistemleri çok işe yarıyor.",
        "En çok ChatGPT gibi sohbet botları ilgimi çekiyor. Bir de robotik ile ilgili şeyler ilgimi çekiyor."    ]
}

df = pd.DataFrame(data)

# Metin temizleme fonksiyonu
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

df['Token'] = df['Yanıt'].apply(temizle)

# Geliştirilmiş tema analizi
temalar = []
for tokens in df['Token']:
    tema_listesi = []
    
    if any(k in tokens for k in ["chatgpt", "sohbet", "öneri", "youtube", "bot", "sohbet botları", "öneri sistemleri"]):
        tema_listesi.append("Sohbet botları ve öneri sistemleri")
    
    if any(k in tokens for k in ["robotik", "otomasyon", "simülasyon", "enerji", "optimizasyon", "tahmin", "veri", "görüntü", "analiz", "makine", "öğrenme", "model", "sistem"]):
        tema_listesi.append("Teknik ve endüstri uygulamaları")
    
    if any(k in tokens for k in ["inşaat", "planlama", "gıda", "kalite", "üretim", "laboratuvar", "hava", "su", "atık", "çevre"]):
        tema_listesi.append("Alan özel uygulamalar")
    
    if any(k in tokens for k in ["bilmiyorum", "ilgim yok", "açıkçası", "fazla", "eğlenceli"]):
        tema_listesi.append("Sınırlı ilgi veya belirsizlik")
    
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df['Temalar'] = temalar

print("Soru 7: İlgi Duyulan Yapay Zekâ Uygulamaları - Tematik Analiz")
print("=" * 80)

# Sonuçları göster
result_df = df[['Katılımcı','Yanıt','Temalar']]
print(result_df.to_string(index=False))

# Tema frekanslarını hesapla
from collections import Counter
print("\n" + "=" * 80)
print("TEMA FREKANSLARI")
print("=" * 80)

tema_frekans = Counter()
for tema in df['Temalar']:
    temas = [t.strip() for t in tema.split(",")]
    for t in temas:
        if t != "Tema bulunamadı":
            tema_frekans[t] += 1

for tema, sayi in tema_frekans.most_common():
    print(f"{tema}: {sayi} katılımcı")

# Bölümlere göre tema dağılımı
print("\n" + "=" * 80)
print("KATILIMCI BAZINDA DETAYLI ANALİZ")
print("=" * 80)

for _, row in df.iterrows():
    print(f"\n{row['Katılımcı']}:")
    print(f"Yanıt: {row['Yanıt']}")
    print(f"Temalar: {row['Temalar']}")
    print("-" * 60)

# Özet istatistikler
print("\n" + "=" * 80)
print("ÖZET İSTATİSTİKLER")
print("=" * 80)
print(f"Toplam katılımcı sayısı: {len(df)}")
print(f"En popüler tema: {tema_frekans.most_common(1)[0][0] if tema_frekans else 'Yok'}")
print(f"En popüler tema katılımcı sayısı: {tema_frekans.most_common(1)[0][1] if tema_frekans else 0}")