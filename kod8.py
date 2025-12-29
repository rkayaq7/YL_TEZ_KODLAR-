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

data8 = {
    "Katılımcı": ["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10"],
    "Yanıt": [
        "Bence çok önemli. Büyük şehirlerde ve çevre danışmanlığı yapan firmalarda veri analizi ve tahminler yapay zekâ ile yönetiliyor. Bu sistemleri anlayabilmek ve yorumlayabilmek büyük avantaj sağlar. Gelecekte çevre mühendislerinin hem sahada hem veri analizinde yapay zekâyı kullanabilmesi gerekecek.",
        "Bence biraz önemli olabilir ama tam emin değilim. İş yerinde bazı makineler yapay zekâ ile çalışıyor ama ben anlamıyorum. Yani belki faydası olur ama şimdilik kafamı kurcalamıyor.",
        "Bence çok önemli. Özellikle elektrik ve elektronik alanında birçok sistem artık yapay zekâ destekli çalışıyor. Otomasyon sistemleri, robotik devreler, veri analizleri gibi işler artık yapay zekâ ile daha verimli hale geliyor. Bir mühendisin bu sistemleri anlayabilmesi ve yorumlayabilmesi büyük avantaj sağlar.",
        "Çok çok önemli olacağını düşünüyorum. Çünkü endüstri mühendislerinin temel görevi sistemi daha verimli yapmak. Artık birçok şirkette üretim planlama ve lojistik sistemlerinde yapay zekâ destekli yazılımlar kullanılıyor. Bir mühendisin bu sistemlerin nasıl çalıştığını bilmesi büyük avantaj olur.",
        "Bence oldukça önemli olacak. Çünkü birçok büyük firma üretim süreçlerini otomatikleştiriyor. Eskiden tamamen insan gözüyle yapılan kalite kontrol artık kameralarla yapılıyor. O kameraların nasıl çalıştığını bilmek bile bir mühendise artı yazar. Ayrıca raporlama, veri analizi gibi şeylerde de yapay zeka çok yardımcı olabilir.",
        "Bence oldukça önemli olacak. Özellikle büyük firmalar artık projelerini dijital sistemler üzerinden yönetiyor ve bazı işlerde yapay zekâ destekli yazılımlar kullanıyor. Bu yazılımları anlayabilmek ve yorumlayabilmek avantaj sağlar.",
        "Bence çok önemli. Büyük gıda firmalarında üretim hatları ve laboratuvar analizleri yapay zekâ ile yönetiliyor. Bu sistemleri anlayabilmek ve gerektiğinde müdahale edebilmek büyük avantaj sağlayacak.",
        "Kesinlikle çok önemli. Şu an bile birçok inşaat şirketi veri analizini bilen mühendisler arıyor. Proje yönetiminde tahmin modelleri, maliyet analizleri, risk yönetimi gibi konularda yapay zeka çok iş görüyor. Bu yüzden temelde bile olsa yapay zekadan anlamak bence önemli bir avantaj.",
        "Bence çok önemli. Özellikle büyük üretim firmalarında robotik sistemler, otomasyon ve veri analizi yapay zekâ ile yönetiliyor. Bu sistemleri anlayabilmek ve gerektiğinde müdahale edebilmek mühendisler için büyük bir avantaj.",
        "Bence önemli olabilir ama tam emin değilim. Şimdiden bazı şirketlerde makineler kendi başına çalışıyormuş gibi, onları anlayabilmek lazım. Ama ben daha 1. sınıfım, tam olarak anlayamadım."    ]
}

df8 = pd.DataFrame(data8)
df8['Token'] = df8['Yanıt'].apply(temizle)

# Geliştirilmiş tema analizi
temalar = []
for tokens in df8['Token']:
    tema_listesi = []
    
    if any(k in tokens for k in ["önemli", "avantaj", "gerekecek", "artı", "fayda", "yararı", "kazanç", "gerekecek", "lazım", "gerekli"]):
        tema_listesi.append("Gelecekte iş hayatında avantaj")
    
    if any(k in tokens for k in ["otomatik", "veri", "analiz", "sistem", "robotik", "otomasyon", "yazılım", "dijital", "teknoloji", "program", "model", "tahmin"]):
        tema_listesi.append("Teknik beceriler ve veri yönetimi")
    
    if any(k in tokens for k in ["tam", "emin", "şimdilik", "sınıf", "anlayamadım", "kafam", "kurcalamıyor", "belki", "biraz", "olabilir"]):
        tema_listesi.append("Belirsizlik ve öğrenme ihtiyacı")
    
    if any(k in tokens for k in ["firma", "şirket", "işyeri", "sektör", "endüstri", "piyasa", "kariyer", "çalışma"]):
        tema_listesi.append("Sektörel gereklilik")
    
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df8['Temalar'] = temalar

print("Soru 8: Mühendislik Alanında Yapay Zekâ Bilgisi Önemi - Tematik Analiz")
print("=" * 80)

# Sonuçları göster
result_df = df8[['Katılımcı','Yanıt','Temalar']]
print(result_df.to_string(index=False))

# Tema frekanslarını hesapla
from collections import Counter
print("\n" + "=" * 80)
print("TEMA FREKANSLARI")
print("=" * 80)

tema_frekans = Counter()
for tema in df8['Temalar']:
    temas = [t.strip() for t in tema.split(",")]
    for t in temas:
        if t != "Tema bulunamadı":
            tema_frekans[t] += 1

for tema, sayi in tema_frekans.most_common():
    print(f"{tema}: {sayi} katılımcı")

# Detaylı analiz
print("\n" + "=" * 80)
print("KATILIMCI BAZINDA DETAYLI ANALİZ")
print("=" * 80)

for _, row in df8.iterrows():
    print(f"\n{row['Katılımcı']}:")
    print(f"Yanıt: {row['Yanıt']}")
    print(f"Temalar: {row['Temalar']}")
    print("-" * 60)