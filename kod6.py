# Soru 6 için tematik analiz 
import pandas as pd
import re
import matplotlib.pyplot as plt
from collections import Counter

# metin işleme için NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NLTK paketlerini sessizce indir
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Türkçe stopword seti
stop_words = set(stopwords.words('turkish'))

# --- 1) Veri: Soru 6 yanıtları ---
data = {
    "Katılımcı": ["K1","K2","K3","K4","K5","K6","K7","K8","K9"],
    "Bölüm": ["Çevre","Elektrik-Elektronik","Endüstri","Gıda","Gıda","İnşaat","İnşaat","Makine","Makine"],
    "Sınıf": ["1","4","3","2","3","3","3","3","1"],
    "Cinsiyet": ["E","E","K","E","K","E","K","K","E"],
    "Yanıt": [
        "Doğrudan bir ders yok. Sadece hocalar bazen 'yapay zekâyla şu simülasyon yapılabilir' falan diyor ama ben anlamıyorum.",
        "Doğrudan ders yok. Ama bazı derslerde örnekler veriliyor. Mesela 'Kontrol Sistemleri' dersinde bazı otomatik kontrol uygulamalarında yapay zekâ mantığı anlatıldı. 'Elektrik Devreleri' dersinde simülasyon ve optimizasyon programları kullanılıyor.",
        "Yapay zekâ dersi yok; 'Veri Analitiği' dersinde bazı temel makine öğrenmesi mantıkları anlatıldı. 'Üretim Yönetimi' dersinde tahmin modellerinden bahsedildi.",
        "Doğrudan bir yapay zekâ dersi yok. 'Gıda İşleme' dersinde akıllı kameralar... 'Gıda Mikrobiyolojisi' dersinde bazı araştırmalarda yapay zekâyla sınıflandırma anlatıldı.",
        "Doğrudan bir yapay zekâ dersi yok ama bazı derslerde uygulamalarını görüyoruz. Örneğin 'Gıda Üretim Teknolojileri' ve 'Laboratuvar Analizleri' derslerinde örnekler var.",
        "Doğrudan yapay zeka dersi yok ama 'Veri Analizi','Python Programlama' ve 'Bilgisayar Destekli Tasarım' gibi dersler alıyoruz.",
        "Doğrudan bir yapay zekâ dersi yok. 'Yapı Malzemesi Laboratuvarı' ve 'İnşaat Proje Yönetimi' derslerinde analiz ve simülasyon örnekleri vardı.",
        "Doğrudan bir yapay zekâ dersi almadım, ama 'Robotik Sistemler' ve 'Üretim Optimizasyonu' derslerinde uygulamalar kullanıldı.",
        "Şu ana kadar doğrudan bir ders almadım. Bazı derslerde hocamız bir örnek veriyor ama detay yok." ]
}

df = pd.DataFrame(data)

# --- 2) Temalar ve anahtar kelimeler ---
themes_keywords = {
    "Doğrudan ders yok": ["doğrudan bir yapay", "doğrudan ders yok", "resmi bir ders yok", "doğrudan bir ders", "resmi eğitim yok", "ders almadım", "ders yok"],
    "Ders içi örneklerle farkındalık": ["ders", "örnek", "anlatıldı", "örnekleri", "örnekler", "hoca", "dersinde", "bahsedildi", "veriliyor"],
    "Uygulama / laboratuvar": ["laboratuvar", "simülasyon", "robotik", "üretim", "kalite", "simülasyon program", "optimizasyon", "kamera", "analiz cihaz", "programlama", "tasarım"],
    "Yüzeysel / sınırlı bilgi": ["kısmi", "tam olarak", "anlamıyorum", "duyduğum", "detay yok", "yüzeysel", "sadece", "bazen"],
    "İlgi artışı / merak ama eksik eğitim": ["ilgi", "merak", "ilgin", "arttı", "merak ediyorum", "ilgi çekti"]
}

# normalize keywords to lowercase (for matching)
for k in themes_keywords:
    themes_keywords[k] = [kw.lower() for kw in themes_keywords[k]]

# --- 3) Tematik eşleştirme fonksiyonu ---
def clean_and_tokens(text):
    if not isinstance(text, str) or text == "":
        return [], ""
    
    try:
        text = text.lower()
        # remove punctuation (basit)
        text = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", " ", text)
        tokens = word_tokenize(text)
        # remove stopwords and single-character tokens (optionally)
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return tokens, text
    except:
        # NLTK hatası durumunda basit yöntem
        text = text.lower()
        text = re.sub(r"[^\w\sçğıöşüÇĞİÖŞÜ]", " ", text)
        tokens = text.split()
        tokens = [t for t in tokens if t not in stop_words and len(t) > 1]
        return tokens, text

def assign_themes(text):
    tokens, lowered = clean_and_tokens(text)
    matched = set()
    
    # 1) match by exact keyword phrases first (safer)
    for theme, kws in themes_keywords.items():
        for kw in kws:
            # phrase match on lowered text
            if re.search(r"\b" + re.escape(kw) + r"\b", lowered):
                matched.add(theme)
    
    # 2) fallback: match by presence of significant tokens (optional)
    token_str = " ".join(tokens)
    fallback_map = {
        "Uygulama / laboratuvar": ["simulasyon", "robotik", "üretim", "laboratuvar", "optimizasyon", "analiz", "programlama", "tasarım"],
        "Ders içi örneklerle farkındalık": ["ders", "hoca", "anlat", "örnek", "bahsedil", "veril"],
        "Yüzeysel / sınırlı bilgi": ["anlamıyorum", "detay", "duydu", "kısmi", "sadece", "bazen"],
        "İlgi artışı / merak ama eksik eğitim": ["merak", "ilgi", "ilgin"],
    }
    
    for theme, kws in fallback_map.items():
        if any(k in token_str for k in kws):
            matched.add(theme)
    
    # ensure "Doğrudan ders yok" if clear negatives present
    if re.search(r"\b(doğrudan|resmi)\b.*\b(ders|eğitim)\b|\bresmi eğitim yok\b|\bdoğrudan ders yok\b|\byok\b.*\bders\b", lowered):
        matched.add("Doğrudan ders yok")
    
    return ", ".join(sorted(matched)) if matched else "Belirsiz / Diğer"

# --- 4) Temaları ata ve tabloyu oluştur ---
df['Temalar'] = df['Yanıt'].apply(assign_themes)

# Göster
print("\n--- Tablo: Soru 6 Tematik Analiz ---\n")
print(df[['Katılımcı','Bölüm','Sınıf','Cinsiyet','Yanıt','Temalar']].to_string(index=False))

# --- 5) Tema frekansları ---
all_themes = []
for t in df['Temalar']:
    for part in [p.strip() for p in t.split(",")]:
        if part and part != "Belirsiz / Diğer":
            all_themes.append(part)

theme_counts = Counter(all_themes)
theme_df = pd.DataFrame.from_records(list(theme_counts.items()), columns=['Tema','Frekans']).sort_values(by='Frekans', ascending=False)

print("\n--- Tema Frekansları ---\n")
print(theme_df.to_string(index=False))

# --- 6) Görselleştirme ---
plt.figure(figsize=(10, 6))
bars = plt.bar(theme_df['Tema'], theme_df['Frekans'], color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3C91E6'])
plt.title('Soru 6: Yapay Zekâ Eğitimi Temaları - Frekans Dağılımı', fontsize=14, fontweight='bold')
plt.xlabel('Temalar')
plt.ylabel('Frekans')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Çubukların üzerine değerleri yaz
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')

plt.show()

# --- 7) Bölümlere göre tema dağılımı ---
print("\n--- Bölümlere Göre Tema Dağılımı ---\n")
bolum_tema = df.groupby(['Bölüm', 'Temalar']).size().unstack(fill_value=0)
print(bolum_tema)