import pandas as pd
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

# NLTK verilerini indir
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

stop_words = set(stopwords.words('turkish'))

# Katılımcı yanıtları
data = {
    "Katılımcı": ["K1","K2","K3","K4","K5","K6","K7","K8","K9","K10"],
    "Yanıt": [
        "Bence yapay zekâ hayatı daha güvenli ve verimli hale getirecek. İnsanların yaptığı rutin işleri üstlenecek ve insanlar daha stratejik veya yaratıcı işlere odaklanabilecek. Çevre mühendisliği açısından, yapay zekâ hava kirliliğini tahmin edebilir, su kaynaklarını izleyebilir ve enerji kullanımını optimize edebilir. Ayrıca sağlık, ulaşım ve şehir planlamasında da büyük etkiler yaratacak. Bazı iş alanlarında dönüşüm olabilir ama genel olarak faydası zararından fazla olacak.",
        "Bence insanlar biraz tembel olabilir, çünkü yapay zekâ bazı işleri yapacak. Ama bazı işler de kaybolabilir. Mesela belki bir gün robotlar çöp toplar, biz hiç uğraşmayız falan. Ama tam olarak nasıl olacak bilmiyorum, kafam karışıyor.",
        "Bence hayatı ciddi şekilde hızlandıracak ve kolaylaştıracak. İnsanların yaptığı rutin işleri üstlenecek ve böylece insanlar daha yaratıcı işlere odaklanabilecek. Elektrik-elektronik alanında, robotik ve otomasyon sistemlerinin yapay zekâyla daha akıllı hale gelmesi işleri daha hızlı ve güvenli yapabilir. Sağlık, eğitim, ulaşım gibi alanlarda da büyük etkiler olabilir. Örneğin otomatik araçların yaygınlaşması, üretim hatlarında verimliliğin artması gibi. Bazı iş alanlarında kayıplar olabilir ama bence yeni iş fırsatları da ortaya çıkacak. Genel olarak bakarsam, insan hayatını daha verimli hale getireceğini düşünüyorum.",
        "Bence hayatı ciddi şekilde hızlandıracak ve sadeleştirecek. İnsanların yaptığı bazı işleri tamamen devralabileceğini düşünüyorum ama bu durum biraz da işin türüne göre değişir. Mesela çok tekrarlayan işler kesinlikle yapay zekaya geçebilir. Ama tamamen insan sezgisi gerektiren işler biraz daha insanda kalır. Ayrıca sağlık, ulaşım, finans gibi alanlarda çok büyük gelişmeler olacağını düşünüyorum. Otomatik çevirilerin bile ne kadar geliştiğini görünce insan ister istemez etkileniyor. Evet, bazı işlerde kaymalar olabilir ama bence yeni iş alanları da doğacak. İnsanlık her teknolojik dönüşümde bunu yaşadı zaten. En büyük etki bence insanlara daha fazla zaman kazandırması olacak.",
        "Bence hayatı ciddi anlamda değiştirmeye devam edecek. Mesela şu an yapay zekâ çok 'yardımcı' rolünde. İnsanların yaptığını hızlandırıyor, kolaylaştırıyor. Ama gelecekte belki daha bağımsız kararlar verecek seviyeye gelebilir. Örneğin sağlık alanında doktorların işini tamamen elinden almaz ama teşhis koymada yüzde 80-90 doğrulukla yardımcı olabilir. Yaşlı bakımında robotlar olabilir. Günlük yaşamda ise bence insanların zaman kazanmasını sağlayacak. Mesela bulaşık makinesi veya çamaşır makinesi ilk çıktığında insanlar ne kadar zaman kazandıklarını fark etmişler. Yapay zekâ da benzer etkiyi bence daha büyük ölçekte yapacak. Tabii iş konusunda bazı değişiklikler olabilir ama bu her teknolojide oldu. Örneğin eskiden elle yapılan şeyler otomatik makineler gelince değişti. Yani ben çok korkmuyorum, çünkü bence insanlar yine yeni işlere yönelir.",
        "Bence insanların günlük işlerini çok kolaylaştıracak. Özellikle gıda sektöründe, üretim hatları daha hızlı ve hatasız çalışacak. Kalite kontrol, stok yönetimi ve tüketici taleplerinin analizi yapay zekâ ile çok daha verimli olacak. Ayrıca sağlık açısından da etkili olabilir; yapay zekâ, besinlerin içerik analizini ve potansiyel alerjenleri hızlı şekilde tespit edebilir. Bunun dışında restoran veya market uygulamalarında kişiye özel öneriler sunarak yaşam kalitesini artırabilir.",
        "Bence hayatımızı çok hızlandıracak ve kolaylaştıracak. İnsanların yaptığı bazı rutin işleri üstlenecek, böylece insanlar daha yaratıcı veya stratejik işlere odaklanabilecek. Örneğin inşaat alanında planlama ve malzeme takibi gibi işlerde yapay zekânın kullanılması işleri hızlandırabilir. Ayrıca sağlık, ulaşım ve eğitim alanlarında da faydalı olacağını düşünüyorum. Tabii bazı iş alanlarında kayıplar olabilir ama bence yeni işler ve fırsatlar da çıkacak. Genel olarak hayatı daha verimli ve kolaylaştırıcı bir şekilde etkileyeceğini düşünüyorum.",
        "Bence büyük bir dönüşüm olacak. Birçok işin daha hızlı yapılacağını ve insanların rutin işlerden biraz daha kurtulacağını düşünüyorum. Ama bunun yanında bazı mesleklerin kaybolacağı ya da çok değişeceği de kesin. İnsanların kendini sürekli geliştirmesi gerekecek. Ama aynı zamanda sağlık, ulaşım ve mühendislik gibi alanlarda büyük fırsatlar yaratacağını düşünüyorum.",
        "Bence günlük hayatımızda çok daha görünür hale gelecek ve işleri hızlandıracak. İnsanların yaptığı tekrarlayan işleri üstlenecek ve insanları daha yaratıcı veya stratejik işlere yönlendirecek. Özellikle makine mühendisliği açısından, üretim hatlarında kalite kontrol, malzeme optimizasyonu ve enerji tasarrufu gibi konularda yapay zekânın etkisi büyük olacak. Bunun dışında sağlık, ulaşım ve eğitim alanlarında da verimliliği artıracak ve insan hatasını azaltacak. Tabii bazı işlerde dönüşüm olabilir ama genel olarak avantajlı olacağını düşünüyorum.",
        "Bence insanlar daha az çalışacak gibi olabilir. Ama tam emin değilim. Mesela otomatik arabalar olabilir, fabrikalarda makineler insan yerine çalışabilir... ya da bazı işler kaybolabilir. Ama bence bazı işler yine insanlara kalır, her şeyi yapay zekâ yapamaz. Yani genel olarak hayatı kolaylaştırabilir ama bazı riskler de olabilir diye düşünüyorum."]
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

df['Token'] = df['Yanıt'].apply(temizle)

# Temaları basit kurallarla çıkarma
temalar = []
for tokens in df['Token']:
    tema_listesi = []
    if any(kelime in tokens for kelime in ["günlük", "hayat", "zaman", "hız", "kolay", "verimli", "güvenli", "sadeleştirecek"]):
        tema_listesi.append("Günlük yaşam ve verimlilik")
    if any(kelime in tokens for kelime in ["rutin", "iş", "meslek", "süreç", "dönüşüm", "kaybolma", "risk", "tembel", "kaybolabilir", "değişecek"]):
        tema_listesi.append("İş süreçleri dönüşümü / olası riskler")
    if any(kelime in tokens for kelime in ["sağlık", "ulaşım", "eğitim", "enerji", "planlama", "kontrol", "optimizasyon", "mühendislik", "üretim", "kalite", "robotik"]):
        tema_listesi.append("Mesleki ve teknik uygulamalar")
    if any(kelime in tokens for kelime in ["yaratıcı", "stratejik", "fırsat", "avantaj", "geliştirmesi", "yeni"]):
        tema_listesi.append("Yeni fırsatlar ve olumlu etkiler")
    
    temalar.append(", ".join(tema_listesi) if tema_listesi else "Tema bulunamadı")

df['Temalar'] = temalar

# Tabloyu göster
print("\nSoru 4 Tematik Analiz Tablosu:\n")
print(df[['Katılımcı', 'Yanıt', 'Temalar']])

# Tema dağılımını göster
from collections import Counter
print("\nTema Dağılımı:")
tema_frekans = Counter([tema for sublist in [t.split(', ') for t in df['Temalar']] for tema in sublist])
for tema, sayi in tema_frekans.items():
    print(f"{tema}: {sayi}")