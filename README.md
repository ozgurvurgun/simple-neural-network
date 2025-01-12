# Simple Neural Network 

Bu proje, 0–9 arasındaki sayıların doğru bir şekilde tahmin edilmesi için bir yapay sinir ağı modelini kullanır. Aynı zamanda yapay sinir ağlarının temellerini anlamak ve uygulamak isteyenler için oluşturulmuş bir başlangıç rehberidir.

## İçerik Rehberi

### **Teorik Bilgiler ve Kavramlar**

Yapay sinir ağlarının temel kavramları ve matematiksel prensiplerini öğrenmek için aşağıdaki belgeye göz atabilirsiniz:

- **Dosya:** [`concepts_and_theory.md`](./concepts_and_theory.md)
- **İçerik:**
  - Nöronlar, katmanlar, ağırlıklar, bias ve aktivasyon fonksiyonları gibi temel kavramlar.
  - Geri yayılım algoritması ve hata fonksiyonları.
  - Model kavramının detaylı açıklamaları ve örnek matematiksel işlemler.

## Projeyi Çalıştırma Adımları

### Sinir Ağının Oluşturulması

`NeuralNetwork.js` dosyasında sinir ağı, aşağıdaki bileşenlere ayrılmıştır:

- **Girdi Katmanı:** 10 nöron.
- **Gizli Katman:** 16 nöron.
- **Çıktı Katmanı:** 10 nöron.

### Modelin Eğitilmesi

`train/train.js` dosyasında sinir ağı eğitimi gerçekleştirilir. Eğitime yön verebilmek için aşağıdaki parametrelerde değişiklik yapabilirsiniz:

#### **Değiştirilebilir Parametreler:**

1. **Girdi ve Hedef Verileri:**

   - `inputs` ve `targets` değişkenleri, modelin hangi veriler üzerinde eğitileceğini belirtir.
   - Örnek:
     ```javascript
     const inputs = [
       [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 0
       [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], // 1
       // Diğer veriler...
     ];
     const targets = inputs; // Hedef veriler, girişlere eşit olabilir.
     ```

2. **Ağ Yapısı:**

   - `NeuralNetwork` sınıfındaki katman yapılandırması düzenlenebilir.
   - Örnek:
     ```javascript
     const nn = new NeuralNetwork(10, 16, 10); // Girdi: 10, Gizli: 16, Çıktı: 10
     ```
     - İlk parametre: Girdi katmanındaki nöron sayısı.
     - İkinci parametre: Gizli katmandaki nöron sayısı.
     - Üçüncü parametre: Çıktı katmanındaki nöron sayısı.

3. **Öğrenme Oranı:**

   - `Trainer` sınıfındaki öğrenme oranı düzenlenebilir.
   - Örnek:
     ```javascript
     const trainer = new Trainer(nn, 0.1); // Öğrenme oranı: 0.1
     ```

4. **Eğitim Döngüsü:**

   - Eğitim tekrar sayısını değiştirebilirsiniz.

   - Daha yüksek tekrar sayısı, modelin girdilere daha iyi uyum sağlamasını sağlar. Bu projede kullanılan veri kümesi oldukça basit ve giriş-çıkış birebir eşleştiği için aşırı öğrenme (overfitting) riski yoktur. Eğitim tekrar sayısını artırmak, genellikle modelin güvenilirliğini artırır.

   - Örnek:

     ```javascript
     trainer.train(inputs, targets, 10000); // 10.000 iterasyon
     ```

### Modelden Çıktı Alma (Inference)

`run/run.js` dosyasında model tahmin işlemi gerçekleştirilir. aşağıdaki bölümlerde belirtilen alanlarda değişiklikler yapabilirsiniz:

#### **Değiştirilebilir Parametreler:**

1. **Girdi Verileri:**

   - `inputVectors` değişkeni, hangi girdiler üzerinde tahmin yapılacağını belirtir.
   - Örnek:
     ```javascript
     const inputVectors = {
       0: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0], // 0
       1: [0, 1, 0, 0, 0, 0, 0, 0, 0, 0], // 1
       // Diğer girdiler...
     };
     ```

2. **Tahmin Edilecek Veri:**

   - `runInference` fonksiyonu içinde tahmin yapılacak giriş vektörü belirtilir.
   - Örnek:
     ```javascript
     runInference(nn, inputVectors[9]); // 9 rakamını tahmin et
     ```

#### **Loglama ve Uyarılar:**

- **Loglama Fonksiyonları:**
  - `logClassProbabilities`: Sınıf olasılıklarını gösterir.
  - `logPredictionResult`: Tahmin edilen sınıfı ve olasılığı gösterir.
- **Uyarılar:**
  - Tahmin edilen olasılık 0.8'in altındaysa, tahminin güvenilir olmayabileceğine dair bir uyarı verir.


## Nasıl Başlanır?

1. **Depoyu Klonlayın:**

   ```bash
   git clone https://github.com/ozgurvurgun/simple-neural-network.git
   ```

2. **Depo Klasörüne Geçin:**

   ```bash
   cd simple-neural-network
   ```

3. **Bağımlılıkları Yükleyin:**

   ```bash
   npm install
   ```

4. **Modeli Eğitin:**

   ```bash
   npm run train
   ```

5. **Modeli Çalıştırın:**

   ```bash
   npm run start
   ```

**Belgelere Göz Atın:**
Önce teorik bilgi için [`concepts_and_theory.md`](./concepts_and_theory.md) dosyasını okuyun, ardından yukarıdaki adımları takip ederek projeyi çalıştırın.

## Geri Bildirim

Herhangi bir sorunuz veya öneriniz varsa lütfen proje deposundaki **Issues** sekmesini kullanarak benimle iletişime geçin.

Keyifli çalışmalar! 🚀

