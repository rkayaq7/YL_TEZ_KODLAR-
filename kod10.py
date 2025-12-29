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
data = {
    "Katılımcı": [
        "Çevre Mühendisliği (K)", "Çevre Mühendisliği (E)", "Elekt.-Elektr. Mühendisliği (E)",
        "Endüstri Mühendisliği (K)", "Gıda Mühendisliği (E)", "Gıda Mühendisliği (K)",
        "İnşaat Mühendisliği (K)", "İnşaat Mühendisliği (E)", "Makine Mühendisliği (K)",
        "Makine Mühendisliği (E)"
    ],
    "Yanıt": [
        "Veri gizliliği çok önemli. İnsanlar hakkında çok fazla veri toplanıyor ve kötüye kullanılabilir. Ayrıca iş kaybı riski var; bazı işler yapay zekâ ile otomatik hale gelebilir. Ama insanlar hâlâ yaratıcı ve kritik kararları verecek. Yapay zekânın sorumlu ve güvenli kullanılması şart.",
        "Bilmiyorum, tam olarak anlamadım. Ama veri gizliliği olabilir sanırım, bir şeyler çalıyor falan diyorlar. İş kaybı da olabilir, robotlar bazı işleri yaparsa insanlar işsiz kalabilir. Onu anladım ama detayını bilmiyorum.",
        "Etik konular çok önemli. Özellikle veri gizliliği ve hatalı kararlar risk yaratabilir. Örneğin bir otomasyon sistemi yanlış karar verirse ciddi sonuçlar doğurabilir. Ama doğru yönetilirse faydası çok daha fazla olur. İnsan gözetimiyle birlikte kullanıldığında, daha güvenli ve verimli bir sistem yaratılabilir. Benim için önemli olan, yapay zekânın sorumlu ve dikkatli bir şekilde kullanılması.",
        "Veri gizliliği konusu beni en çok düşündüren şey. Sürekli bir şeyleri takip eden sistemler var ve bunların nasıl kullanıldığı tam olarak açıklanmıyor. İnsanların tercihlerine göre içerik gösterilmesi bir yere kadar iyi ama bazen manipülatif olduğunu da hissediyorum. Ayrıca yapay zekanın hatalı karar vermesi de bir risk.",
        "Etik konular bence çok önemli. Bir kere veri gizliliği konusu çok hassas. Çünkü sosyal medya uygulamaları kimin ne izlediğini biliyor, bu biraz tedirgin edici aslında. Bir de yapay zeka yanlış bilgi verirse sorun olabilir. Özellikle sağlık gibi alanlarda kesinlikle kontrol edilmesi lazım.",
        "Veri gizliliği önemli; özellikle tüketici verileri toplanıyor ve kötüye kullanılabilir. Ayrıca iş kaybı riski var; bazı rutin işler makineler tarafından üstlenilebilir. Ama insanlar hâlâ kritik kararları verecek, bu yüzden yapay zekâ sorumlu ve bilinçli şekilde kullanılmalı.",
        "Bence etik çok önemli bir konu. Veri gizliliği, iş kaybı ve hatalı kararlar en çok düşündüğüm şeyler. Örneğin şantiye planlamasında bir hata oluşursa sonuçları ciddi olabilir. Ama doğru yönetilirse faydası daha fazla olur. Benim için önemli olan hem güvenli hem de adil bir şekilde kullanılması.",
        "Bence en büyük sorun veri gizliliği. Hangi verilerin kullanıldığı belli değil. Bir de iş kaybı konusu var; bazı insanların bu teknolojiler yüzünden işsiz kalması endişe verici. Ayrıca yapay zekanın yanlış karar vermesi durumunda sorumluluğun kimde olacağı konusu tam net değil. Bu yüzden etik kuralların daha da geliştirilmesi gerektiğini düşünüyorum.",
        "Bence veri gizliliği çok önemli. İnsanlar hakkında çok fazla veri toplandığı için kötü amaçla kullanılabilir. Ayrıca iş kaybı riski var; bazı işleri makineler üstlenebilir. Ama insanlar hâlâ yaratıcı ve kritik kararları verecek. Bu yüzden yapay zekâ hem sorumlu hem güvenli kullanılmalı.",
        "Bence veri gizliliği önemli. Mesela sosyal medya benim hakkımda çok şey biliyor, bu biraz tedirgin edici. Bir de bazı işleri insanlar yerine yaparsa iş kaybı olabilir. Ama çok emin değilim, tam olarak anlamadım. Yani bazı riskler olabilir diye düşünüyorum ama detayını bilmiyorum." ]
}

df = pd.DataFrame(data)

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

df['Token'] = df['Yanıt'].apply(temizle)

# Geliştirilmiş tematik kodlama
temalar = []
for tokens in df['Token']:
    tema_listesi = []
    
    if any(k in tokens for k in ["veri", "gizlilik", "kişisel", "gizli", "mahrem", "tedirgin", "takip", "izleme"]):
        tema_listesi.append("Veri gizliliği")
    
    if any(k in tokens for k in ["iş", "kaybı", "işsiz", "işsizlik", "istihdam", "çalışma", "meslek"]):
        tema_listesi.append("İş kaybı riski")
    
    if any(k in tokens for k in ["hatalı", "yanlış", "risk", "hata", "yanlış karar", "tehlike", "sorun"]):
        tema_listesi.append("Hatalı karar riski")
    
    if any(k in tokens for k in ["sorumlu", "güvenli", "kontrol", "etik", "dikkatli", "güvenlik", "adil", "sorumluluk"]):
        tema_listesi.append("Sorumlu ve güvenli kullanım")
    
    if any(k in tokens for k in ["bilmiyorum", "anlamadım", "emin değilim", "tam olarak", "detay", "sanırım"]):
        tema_listesi.append("Belirsizlik veya bilgi eksikliği")
    
    if any(k in tokens for k in ["manipülatif", "kötüye kullanım", "kötü amaç", "kötüye", "yanlış bilgi"]):
        tema_listesi.append("Kötüye kullanım riski")
    
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df['Temalar'] = temalar

print("Soru 9: Yapay Zekâ ile İlgili Endişe ve Etik Konular - Tematik Analiz")
print("=" * 90)

# Sonuçları göster
result_df = df[['Katılımcı','Yanıt','Temalar']]
print(result_df.to_string(index=False))

# Tema frekanslarını hesapla
from collections import Counter
print("\n" + "=" * 90)
print("TEMA FREKANSLARI")
print("=" * 90)

tema_frekans = Counter()
for tema in df['Temalar']:
    temas = [t.strip() for t in tema.split(",")]
    for t in temas:
        if t != "Tema bulunamadı":
            tema_frekans[t] += 1

for tema, sayi in tema_frekans.most_common():
    print(f"{tema}: {sayi} katılımcı")

# Detaylı analiz
print("\n" + "=" * 90)
print("KATILIMCI BAZINDA DETAYLI ANALİZ")
print("=" * 90)

for _, row in df.iterrows():
    print(f"\n{row['Katılımcı']}:")
    print(f"Yanıt: {row['Yanıt']}")
    print(f"Temalar: {row['Temalar']}")
    print("-" * 80)

# Özet istatistikler
print("\n" + "=" * 90)
print("ÖZET İSTATİSTİKLER")
print("=" * 90)
print(f"Toplam katılımcı sayısı: {len(df)}")
print(f"En yaygın endişe: {tema_frekans.most_common(1)[0][0] if tema_frekans else 'Yok'}")
print(f"En yaygın endişe katılımcı sayısı: {tema_frekans.most_common(1)[0][1] if tema_frekans else 0}")