import pandas as pd
import re
# Görüşme yanıtları
responses = [
    "Öncelikle güvenilir ve doğru bilgi sunmasını isterim, özellikle veri analizi ve çevre tahminlerinde hata yapmaması önemli.",
    "Öncelikle yanlış bilgi vermemesini isterim, çünkü bazen saçmalıyor. Sonra belki eğlenceli olur, bazı şeyleri otomatik yapar, bana zaman kazandırır.",
    "Öncelikle güvenilir olmasını isterim. Ayrıca kişisel olması güzel olur. Örneğin görsel öğreniyorsam diyagram veya basit şemalar ile açıklama yapması iyi olur. Son olarak, kaynak göstermesi bence önemli.",
    "Öncelikle daha güvenilir olmasını isterim. Ayrıca daha kişisel hale gelmesini isterim. Son olarak, kaynak göstermesi çok önemli.",
    "Öncelikle güvenilir ve doğru bilgi sunmasını isterim. İkincisi, daha hızlı ve verimli çalışması. Üçüncüsü, kullanıcı dostu ve anlaşılır olması.",
    "Öncelikle daha güvenilir olmasını isterim. Ayrıca daha kişisel olmasını isterim. Bir de bence daha şeffaf olabilir.",
    "Öncelikle güvenilir olmasını isterim. Bir de kişisel olmasını isterim. Son olarak, kaynak göstermesi bence önemli.",
    "Benim beklentim, daha güvenilir ve açıklanabilir bir hale gelmesi. Ayrıca daha kullanıcı dostu hale gelmesini bekliyorum.",
    "İlk olarak, güvenilir ve doğru bilgi sunmasını bekliyorum. İkinci olarak, öğrenme sürecime uygun şekilde yardımcı olmasını isterim. Üçüncü olarak da kaynak göstermesi çok önemli.",
    "Daha doğru ve güvenilir olmasını isterim. Ayrıca biraz daha kolay anlaşılır olmasını isterim." ]
# Temalar ve anahtar kelimeler
themes = {
    "Güvenilirlik": ["güvenilir", "doğru bilgi", "yanlış bilgi", "hata"],
    "Kişiselleştirme": ["kişisel", "öğrenme tarzı", "ilgim", "öğrenme sürecime uygun"],
    "Kaynak/Gösterim": ["kaynak", "referans", "şeffaf"],
    "Hız/Verimlilik": ["hızlı", "verimli", "zaman kazandırır"],
    "Kullanıcı Dostu/Anlaşılır": ["anlaşılır", "kullanıcı dostu", "kolay"]
}
# Tematik analiz tablosu için boş liste
analysis_table = []
# Her yanıt için temaları kontrol et
for i, response in enumerate(responses, 1):
    response_lower = response.lower()
    matched_themes = []
    for theme, keywords in themes.items():
        if any(re.search(r"\b" + kw + r"\b", response_lower) for kw in keywords):
            matched_themes.append(theme)
    analysis_table.append({"Yanıt No": i, "Yanıt": response, "Temalar": ", ".join(matched_themes)})
# Tabloyu DataFrame olarak oluştur
df = pd.DataFrame(analysis_table)
# Tabloyu göster
print(df)
