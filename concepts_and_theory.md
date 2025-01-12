## Yapay Sinir Ağı Nedir?

Bir yapay sinir ağı, insan beyninin çalışma prensiplerini taklit ederek bilgiyi öğrenir ve tahminler yapar. Yapay sinir ağları, veri işlemek ve öğrenmek için biyolojik nöronlardan esinlenerek tasarlanmıştır.

### Temel Kavramlar

#### **Nöron (Neuron)**

Bir nöron, bir sinir ağının temel yapı taşıdır. Aşağıdaki işlemleri gerçekleştirir:

1. **Girdi Almak (Input):** Diğer nöronlardan veya veri kümesinden bilgi alır.
2. **Ağırlıklarla Çarpma (Weighting):** Her bir girdiyi, o girdinin önemini temsil eden bir ağırlık (weight) ile çarpar.
3. **Toplama (Summation):** Tüm ağırlıklı girdileri toplar ve bir "net girdi" hesaplar:
   ```math
   Net = \sum_{i} (Girdi_{i} \cdot Ağırlık_{i}) + Bias
   ```
4. **Aktivasyon Fonksiyonu (Activation Function):** Net girdiyi işleyerek bir çıktı (output) üretir.

#### **Katmanlar (Layers)**

1. **Girdi Katmanı (Input Layer):** Veri sinir ağına bu katman aracılığıyla girer. Bu projede her rakam, 10 elemanlı bir dizi ile temsil edilir.
   - Örneğin: `0` rakamı `[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]` olarak(One Hot Encoding) temsil edilir.
2. **Gizli Katman (Hidden Layer):** Girdi verilerini işleyerek ara çıktıların oluşturulduğu katmandır. Gizli katman sayısı ve nöron sayısı, problemin karmaşıklığına bağlı olarak değişir. Bu projede 16 gizli nöron vardır.
3. **Çıktı Katmanı (Output Layer):** Nihai tahminlerin yapıldığı katmandır. Bu projede 10 çıktı nöronu vardır (0–9 arasındaki rakamları temsil eder).

#### **Ağırlıklar (Weights)**

Ağırlıklar, bir nöronun bir başka nöron üzerindeki etkisini ifade eder. Örneğin, bir girdi verisinin ağdaki bir nörona etkisi ne kadar büyükse, o veriye atanan ağırlık o kadar yüksek olacaktır. Ağırlıklar, eğitim sürecinde güncellenerek optimize edilir.

#### **Bias (Önyargı)**

Bias, bir nöronun aktivasyonunu girdi değerlerinden bağımsız olarak ayarlamak için kullanılan bir sabittir. Bias, ağın daha esnek olmasını sağlar ve yalnızca ağırlıklarla çözülemeyen durumları düzeltir:
```math
b_{yeni} = b_{eski} + (\eta \cdot Hata)
```

#### **Aktivasyon Fonksiyonu (Activation Function)**

Bir yapay sinir ağında aktivasyon fonksiyonları, nöronun aldığı girdilerden anlamlı bir çıktı üretmek için kullanılır. Nöronun temel görevi, aldığı sinyalleri (girdileri) işlemek ve diğer nöronlara veya sistem bileşenlerine aktarabileceği bir değer döndürmektir.

Bu projede **Sigmoid Fonksiyonu** kullanılmıştır:
- **Sigmoid Fonksiyonu:**
  ```math
  f(x) = \frac{1}{1 + e^{-x}}
  ```
  - Çıktıyı 0 ile 1 arasında sınırlar.
  - Çıktılar olasılık olarak yorumlanabilir.
  - Türevi:
    ```math
    f'(x) = f(x) \cdot (1 - f(x))
    ```

#### **Geri Yayılım (Backpropagation)**

Geri yayılım algoritması, sinir ağının çıktılarındaki hataları azaltmak için kullanılan bir yöntemdir. Bu algoritma, ağın ağırlıklarını ve bias değerlerini güncelleyerek daha doğru tahminler yapılmasını sağlar. 

1. **Hata Fonksiyonu (Error Function):**
   - Hata fonksiyonu genellikle ortalama kare hatası (MSE) olarak tanımlanır:
     ```math
     E = \frac{1}{2} \sum_{i} (y_{gerçek}^{(i)} - y_{tahmin}^{(i)})^2
     ```

2. **Ağırlık Güncellemesi:**
   - Ağırlıklar şu şekilde güncellenir:
     ```math
     W_{yeni} = W_{eski} - \eta \cdot \frac{\partial E}{\partial W}
     ```

3. **Bias Güncellemesi:**
   - Bias değerleri benzer bir yöntemle güncellenir:
     ```math
     b_{yeni} = b_{eski} + (\eta \cdot Hata)
     ```

#### **Model**

Bir yapay sinir ağı modeli, öğrenme sürecinde elde edilen bilgilerle parametrelerin (ağırlıklar ve biaslar) yapılandırıldığı bir yapıdır. Model, bir girdiyi alıp işleyerek bir çıktı tahmini yapabilen matematiksel bir fonksiyon olarak düşünülebilir.

#### **Modelin Özellikleri:**

1. **Parametreler:** 
   - Modelin öğrendiği ağırlıklar (\(W\)) ve biaslar (\(b\)).
   - Bu parametreler, modelin giriş ile çıkış arasındaki ilişkileri temsil eder.

2. **Yapı:**
   - Model, katmanlar ve nöronlardan oluşur. Her katman, girdileri işleyerek bir sonraki katmana aktarır.

3. **Eğitim Süreci:**
   - Model, bir eğitim verisi kümesi üzerinde çalışarak parametrelerini optimize eder.
   - Hedef, gerçek çıktılar ile model çıktıları arasındaki farkı (hata) minimize etmektir.

#### **Modelin Önemi:**

Model, sinir ağı tarafından öğrenilen bilginin soyut bir temsili olarak, gerçek dünyadaki problemleri çözmek için kullanılır. Örneğin:
- El yazısı rakamların tanınması,
- Görüntü sınıflandırma,
- Doğal dil işleme görevleri gibi birçok uygulama bu tür modellerle gerçekleştirilir.

#### **Matematiksel Açıklama:**

Modelin bir girdiyi nasıl işlediği şu şekilde özetlenebilir:
```math
y_{tahmin} = f(W \cdot X + B)
```
Burada:
- \(X\): Giriş vektörü,
- \(W\): Ağırlıklar matrisi,
- \(B\): Bias vektörü,
- \(f\): Aktivasyon fonksiyonu,
- \(y_{tahmin}\): Tahmin edilen çıktı.

#### **Model Kaydetme ve Yükleme:**

Eğitilen bir model, daha sonra kullanılmak üzere kaydedilebilir ve farklı bir uygulama veya analiz için yeniden yüklenebilir.

#### **Model Neye Benzer?**

Bir yapay sinir ağı modeli, genellikle ONNX, JSON, YAML,  h5... gibi formatlarda saklanır. Bu tür bir dosyanın içeriği, modelin mimarisi, katmanların ağırlıkları, bias değerleri ve diğer parametreleri içerir.

#### **Bir Model Dosyası Nasıl Görünür?**

Aşağıda, JSON formatında kaydedilmiş basit bir sinir ağı modelinin örneği verilmiştir:

```json
{
  "model_config": {
    "input_layer": {
      "size": 4
    },
    "hidden_layers": [
      {
        "size": 16,
        "activation": "sigmoid"
      }
    ],
    "output_layer": {
      "size": 3,
      "activation": "softmax"
    }
  },
  "weights": {
    "input_to_hidden": [
      [0.12, -0.23, 0.45, 0.67],
      [-0.12, 0.34, -0.56, 0.78],
      ...
    ],
    "hidden_to_output": [
      [0.15, 0.27, -0.36],
      [-0.45, 0.58, 0.61],
      ...
    ]
  },
  "biases": {
    "hidden_layer": [0.01, -0.02, 0.03, ...],
    "output_layer": [0.05, 0.12, -0.15]
  }
}
```

---

## Projede Kullanılan Algoritmalar

### 1. **İleri Besleme (Feedforward)**

Girdi verileri, sinir ağının katmanlarından geçirilerek bir çıktı oluşturulur. Ağırlıklar ve biaslar kullanılarak her katmandaki nöronun aktivasyonu hesaplanır.

1. **Net Girdi Hesaplama:**
   ```math
   Net = \sum_{i} (Girdi_{i} \cdot Ağırlık_{i}) + Bias
   ```
2. **Aktivasyon Hesaplama:**
   ```math
   Çıktı = f(Net)
   ```

### 2. **Geri Yayılım (Backpropagation)**

Bu algoritma, hata fonksiyonunun türevlerini kullanarak ağırlıkları ve biasları günceller.
- **Ağırlık Güncelleme:**
  ```math
  W = W + (Hata \cdot Aktivasyon \cdot Öğrenme Oranı)
  ```

### 3. **Matris İşlemleri**

Ağırlıkların ve girdilerin işlenmesi matris çarpımlarıyla yapılır. Bu, verimli ve hızlı hesaplama sağlar:
```math
Z = W \cdot X + B
```
