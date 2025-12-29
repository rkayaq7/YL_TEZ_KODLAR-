import pandas as pd
import re
from collections import Counter

# Katılımcı yanıtları
yanitlar = {
    "K1": "Valla sanmam. Çok karışık gibi geliyor. Kod yazmak falan lazım, onu hiç sevmem.",
    "K2": "Tam uzmanlaşmak istemem ama temel seviyede öğrenmek isterim.",
    "K3": "Tamamen uzman olmak istemem ama alanımda kullanabilecek seviyede bilgi isterim.",
    "K4": "Uzmanı olmayabilirim ama iş hayatında kullanabilecek düzeyde bilgi isterim.",
    "K5": "Uzmanı olur muyum bilmiyorum ama temel düzeyde bilgi isterim.",
    "K6": "Tam uzman olmak istemem ama gıda alanında uygulama odaklı bilgi isterim.",
    "K7": "Uzmanlaşmak istemem ama inşaat alanında temel düzeyde bilgi isterim.",
    "K8": "İsterim ama tamamen bu alana yönelir miyim emin değilim.",
    "K9": "Uzman olmak istemem ama robotik ve üretim alanı için temel bilgi isterim.",
    "K10": "Şimdilik istemem çünkü çok karışık ama alanımla ilgisi olursa öğrenmek isterim."
}

df = pd.DataFrame(yanitlar.items(), columns=["Katılımcı", "Yanıt"])

# Geliştirilmiş temalar
temalar = {
    "Uzmanlaşmak istememe (Zorluk Algısı)": ["zor", "karışık", "istemem", "kod", "sevmem", "sanmam", "istemez", "istemiyorum"],
    "Temel/Uygulama Odaklı Bilgi Edinme İsteği": ["temel", "kullan", "uygulama", "öğrenmek", "bilgi", "seviye", "düzey", "öğrenmek isterim", "bilgi isterim"],
    "Mesleki Fayda Görme": ["iş hayatı", "avantaj", "fayda", "robotik", "üretim", "sektör", "alanımda", "gıda", "inşaat", "meslek"],
    "Kararsızlık / Şüphe": ["emin değilim", "bilmiyorum", "şimdilik", "olur muyum", "olmayabilirim"]
}

def tema_bul(yazi):
    if not isinstance(yazi, str) or yazi == "":
        return "Tema bulunamadı"
    
    bulunan = []
    yazi_lower = yazi.lower()
    
    for tema, kelimeler in temalar.items():
        for k in kelimeler:
            # Regex ile kelime sınırlarını kontrol ederek arama
            if re.search(r'\b' + re.escape(k) + r'\b', yazi_lower):
                bulunan.append(tema)
                break  # Aynı tema için birden fazla eşleşme olmasın
    
    # Eğer hiç tema bulunamazsa
    if not bulunan:
        return "Tema bulunamadı"
    
    return ", ".join(bulunan)

df["Tema"] = df["Yanıt"].apply(tema_bul)

# Sonuçları göster
print("Soru 5: Yapay Zekâ Uzmanlığı İsteme Durumu - Tematik Analiz")
print("=" * 70)
print(df.to_string(index=False))

# Tema frekanslarını hesapla
print("\n" + "=" * 70)
print("TEMA FREKANSLARI")
print("=" * 70)

tema_frekans = Counter()
for tema in df["Tema"]:
    # Birden fazla tema varsa ayır
    temas = [t.strip() for t in tema.split(",")]
    for t in temas:
        if t != "Tema bulunamadı":
            tema_frekans[t] += 1

for tema, sayi in tema_frekans.most_common():
    print(f"{tema}: {sayi} katılımcı")

# Detaylı analiz
print("\n" + "=" * 70)
print("DETAYLI ANALİZ")
print("=" * 70)

for _, row in df.iterrows():
    print(f"\n{row['Katılımcı']}: {row['Yanıt']}")
    print(f"Tema: {row['Tema']}")

# Özet istatistikler
print("\n" + "=" * 70)
print("ÖZET İSTATİSTİKLER")
print("=" * 70)
print(f"Toplam katılımcı sayısı: {len(df)}")
print(f"Tema bulunan katılımcı sayısı: {len(df[df['Tema'] != 'Tema bulunamadı'])}")
print(f"En yaygın tema: {tema_frekans.most_common(1)[0][0] if tema_frekans else 'Yok'}")