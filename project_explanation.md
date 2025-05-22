# شرح كامل لكود مشروع Research Paper Recommendation 📚💻

## 1️⃣ استيراد المكتبات (Importing Libraries)

```python
import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle
```

**الشرح:**
- **numpy**: للعمليات الحسابية على المصفوفات
- **pandas**: لقراءة ومعالجة البيانات الجدولية
- **matplotlib**: لرسم الرسوم البيانية
- **literal_eval**: لتحويل النصوص لكائنات Python (مثل تحويل "[1,2,3]" إلى قائمة حقيقية)
- **train_test_split**: لتقسيم البيانات
- **tensorflow/keras**: للـ Deep Learning
- **TfidfVectorizer**: لتحويل النص إلى أرقام
- **pickle**: لحفظ النماذج المدربة

---

## 2️⃣ قراءة البيانات (Reading Data)

```python
df = pd.read_csv("arxiv_data_210930-054931.csv")
df.head()
```

**الشرح:**
- بيقرأ ملف CSV يحتوي على بيانات الأوراق البحثية من ArXiv
- `df.head()` يعرض أول 5 صفوف للاطلاع على شكل البيانات

---

## 3️⃣ تنظيف ومعالجة البيانات (Data Cleaning)

### أ) استكشاف البيانات
```python
df['terms']                    # عرض عمود التصنيفات
df['terms'].unique()           # عرض جميع التصنيفات الفريدة
df['terms'].value_counts()[:10] # عرض أكثر 10 تصنيفات تكراراً
```

### ب) إزالة التكرارات
```python
df.duplicated().sum()          # عدد الصفوف المكررة
df.drop_duplicates(inplace=True) # حذف الصفوف المكررة
df.shape                       # شكل البيانات بعد الحذف
```

### ج) تنظيف التصنيفات
```python
df['terms'] = df['terms'].apply(literal_eval)  # تحويل النص إلى قائمة حقيقية
```

**مثال:**
```
قبل: "['cs.AI', 'cs.LG']"      # نص
بعد: ['cs.AI', 'cs.LG']        # قائمة Python حقيقية
```

### د) إزالة التصنيفات النادرة
```python
term_counts = df['terms'].value_counts()       # عد تكرار كل تصنيف
common_terms = term_counts[term_counts > 1].index  # التصنيفات المتكررة أكتر من مرة
df = df[df['terms'].isin(common_terms)]        # الاحتفاظ بالتصنيفات الشائعة فقط
```

**ليه؟** التصنيفات اللي تظهر مرة واحدة بس مش مفيدة للنموذج

---

## 4️⃣ تقسيم البيانات (Data Splitting)

```python
# تقسيم رئيسي: 90% تدريب، 10% اختبار
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df['terms'])

# تقسيم مجموعة الاختبار لنصين
val_df = test_df.sample(frac=0.5, random_state=42)    # 50% للتحقق
test_df = test_df.drop(val_df.index)                  # 50% للاختبار النهائي

print(f"Train size : {len(train_df)}")
print(f"Test size : {len(test_df)}")
print(f"Validation size : {len(val_df)}")
```

**النتيجة النهائية:**
- **Training (90%)**: للتعلم
- **Validation (5%)**: لضبط النموذج أثناء التدريب
- **Test (5%)**: للتقييم النهائي

**stratify=df['terms']**: يضمن أن نسب التصنيفات متشابهة في كل مجموعة

---

## 5️⃣ تحضير التصنيفات (Target Preparation)

```python
from sklearn.preprocessing import MultiLabelBinarizer

# إنشاء وتدريب المحول
mlb = MultiLabelBinarizer()
mlb.fit(train_df['terms'])

# الحصول على قائمة التصنيفات (المفردات)
vocab = mlb.classes_
print(f"Number of classes: {len(vocab)}")  # عدد التصنيفات الفريدة

# تحويل التصنيفات إلى تشفير binary
train_labels = mlb.transform(train_df['terms'])
val_labels = mlb.transform(val_df['terms'])
test_labels = mlb.transform(test_df['terms'])
```

**مثال على التشفير:**
```python
# المدخل الأصلي
sample = ['cs.AI', 'cs.LG']

# بعد التشفير (لو عندنا 5 تصنيفات مثلاً)
binarized = [1, 0, 1, 0, 0]  # 1 يعني التصنيف موجود، 0 يعني مش موجود
```

### دالة عكس التشفير
```python
def invert_multi_hot(encoded_labels):
    return [vocab[i] for i, val in enumerate(encoded_labels) if val == 1]
```

---

## 6️⃣ معالجة النصوص (Text Vectorization)

```python
# إنشاء مجموعة المفردات
vocabulary = set()
train_df['abstracts'].str.lower().str.split().apply(vocabulary.update)
vocab_size = len(vocabulary)

# إعداد محول TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    max_features=min(100000, vocab_size),  # أقصى عدد مفردات
    ngram_range=(1, 2),                    # كلمات مفردة + أزواج كلمات
    stop_words='english',                  # إزالة كلمات مثل (the, a, an)
    min_df=5                               # الكلمة تظهر في 5 وثائق على الأقل
)

# تدريب المحول وتطبيقه
tfidf_vectorizer.fit(train_df['abstracts'])
train_features = tfidf_vectorizer.transform(train_df['abstracts'])
val_features = tfidf_vectorizer.transform(val_df['abstracts'])
test_features = tfidf_vectorizer.transform(test_df['abstracts'])
```

**TF-IDF شرح مبسط:**
- **TF** (Term Frequency): كام مرة الكلمة ظهرت في الوثيقة
- **IDF** (Inverse Document Frequency): قد إيه الكلمة نادرة في كل المجموعة
- **النتيجة**: رقم يوضح أهمية الكلمة للوثيقة دي

**مثال:**
كلمة "neural" في ورقة عن الشبكات العصبية:
- لو ظهرت 5 مرات في الورقة (TF عالي)
- ومش موجودة في كتير ورق تانية (IDF عالي)
- النتيجة: رقم عالي = كلمة مهمة لهذه الورقة

---

## 7️⃣ تحويل البيانات لصيغة TensorFlow

```python
import scipy.sparse as sp
from tensorflow.keras.utils import Sequence

class SparseDataGenerator(Sequence):
    """
    مولد بيانات يحول المصفوفات المتناثرة إلى كثيفة في دفعات صغيرة
    لتجنب مشاكل الذاكرة
    """
    def __init__(self, X_sparse, y, batch_size):
        self.X_sparse = X_sparse
        self.y = y
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(self.X_sparse.shape[0] / self.batch_size))
    
    def __getitem__(self, idx):
        # حساب مؤشرات الدفعة
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, self.X_sparse.shape[0])
        
        # استخراج وتحويل بيانات الدفعة
        X_batch = self.X_sparse[start_idx:end_idx].toarray()
        y_batch = self.y[start_idx:end_idx]
        
        return X_batch, y_batch
```

**ليه المولد ده؟**
- بيانات TF-IDF كبيرة جداً
- تحميلها كلها في الذاكرة ممكن يخلي الكمبيوتر يعلق
- المولد يدي الذاكرة دفعات صغيرة كل مرة

```python
# إنشاء مولدات البيانات
batch_size = 128
train_generator = SparseDataGenerator(train_features, train_labels, batch_size)
val_generator = SparseDataGenerator(val_features, val_labels, batch_size)
test_generator = SparseDataGenerator(test_features, test_labels, batch_size)
```

---

## 8️⃣ بناء النموذج (Model Architecture)

```python
# حساب أبعاد الدخل والخرج
input_dim = train_features.shape[1]  # عدد المفردات من TF-IDF
output_dim = len(vocab)              # عدد التصنيفات

model = keras.Sequential([
    # طبقة الدخل
    layers.Input(shape=(input_dim,)),
    
    # الطبقة المخفية الأولى
    layers.Dense(512),                    # 512 نيورون
    layers.BatchNormalization(),          # تطبيع للتسريع والاستقرار
    layers.Activation("relu"),            # دالة التفعيل
    layers.Dropout(0.5),                  # منع الحفظ الأعمى
    
    # الطبقة المخفية الثانية
    layers.Dense(256),                    # 256 نيورون (أقل من السابقة)
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    
    # الطبقة المخفية الثالثة
    layers.Dense(128),                    # 128 نيورون
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    
    # الطبقة المخفية الرابعة
    layers.Dense(64),                     # 64 نيورون
    layers.BatchNormalization(),
    layers.Activation("relu"),
    layers.Dropout(0.5),
    
    # طبقة الخرج
    layers.Dense(output_dim, activation='sigmoid')  # Sigmoid للتصنيف المتعدد
])
```

**شرح مكونات النموذج:**

### Dense Layers (الطبقات الكثيفة)
- كل نيورون متصل بكل نيورونات الطبقة السابقة
- الأرقام (512, 256, 128, 64) تقل تدريجياً = تركيز المعلومات

### BatchNormalization
- بيطبع المدخلات عشان التدريب يكون أسرع وأكثر استقراراً
- زي ما تقول للنموذج "خلي البيانات منتظمة"

### ReLU Activation
```python
# دالة ReLU بسيطة
def relu(x):
    return max(0, x)  # لو القيمة سالبة، خليها صفر
```

### Dropout
- بيشيل 50% من الاتصالات عشوائياً أثناء التدريب
- يمنع النموذج إنه يحفظ بدل ما يتعلم

### Sigmoid في الخرج
- بيطلع أرقام بين 0 و 1
- كل رقم يمثل احتمالية إن التصنيف ده موجود

---

## 9️⃣ تدريب النموذج (Model Training)

```python
# إعداد النموذج للتدريب
model.compile(
    loss="binary_crossentropy",     # دالة الخسارة للتصنيف المتعدد
    optimizer='adam',               # محسن سريع وفعال
    metrics=['binary_accuracy']     # مقياس الأداء
)

# الإيقاف المبكر
early_stopping = EarlyStopping(
    patience=5,                     # لو الأداء ما تحسنش لـ 5 epochs
    restore_best_weights=True       # ارجع لأحسن وزن
)

# التدريب
history = model.fit(
    train_generator,                # بيانات التدريب
    validation_data=val_generator,  # بيانات التحقق
    epochs=20,                      # عدد الدورات
    callbacks=[early_stopping]     # الإيقاف المبكر
)
```

**شرح المفاهيم:**

### Binary Crossentropy
- دالة خسارة مناسبة للتصنيف المتعدد
- بتحسب الخطأ بين التوقع والواقع لكل تصنيف

### Adam Optimizer
- محسن ذكي يعدل الأوزان بناءً على الأخطاء
- أسرع وأكثر فعالية من المحسنات التقليدية

### Early Stopping
- لو الأداء على validation توقف عن التحسن لـ 5 epochs
- يوقف التدريب ويرجع لأحسن وزن
- يمنع overfitting (الحفظ الأعمى)

---

## 🔟 رسم النتائج (Metrics Plotting)

```python
def plot_result(item):
    plt.plot(history.history[item], label=item)
    plt.plot(history.history["val_" + item], label="val_" + item)
    plt.xlabel("Epochs")
    plt.ylabel(item)
    plt.title("Train and Validation {} Over Epochs".format(item), fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

plot_result("loss")              # رسم الخسارة
plot_result("binary_accuracy")   # رسم الدقة
```

**الهدف:**
- متابعة أداء النموذج أثناء التدريب
- التأكد إنه مش بيحفظ (لو خط validation بيرتفع والـ training بينزل = overfitting)

---

## 1️⃣1️⃣ تقييم النموذج (Model Evaluation)

```python
# تقييم الأداء على المجموعات المختلفة
train_loss, binary_acc_train = model.evaluate(train_generator)
val_loss, binary_acc_val = model.evaluate(val_generator)
test_loss, binary_acc_test = model.evaluate(test_generator)

print(f"Binary accuracy on the train set: {round(binary_acc_train * 100, 2)}%.")
print(f"Binary accuracy on the validation set: {round(binary_acc_val * 100, 2)}%.")
print(f"Binary accuracy on the test set: {round(binary_acc_test * 100, 2)}%.")
```

**Binary Accuracy إيه؟**
- بيحسب كام تصنيف صح من إجمالي التصنيفات
- مثلاً: لو ورقة فيها 3 تصنيفات والنموذج عرف 2 منهم = 66.7%

---

## 1️⃣2️⃣ حفظ النموذج (Save Model)

```python
# حفظ النموذج المدرب
model.save("subject_area_model.keras")

# حفظ محول النصوص
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# حفظ محول التصنيفات
with open('label_binarizer.pkl', 'wb') as f:
    pickle.dump(mlb, f)
```

**ليه بنحفظ 3 حاجات؟**
1. **النموذج**: الذكاء الاصطناعي المدرب
2. **TF-IDF**: عشان نقدر نحول النص الجديد لأرقام
3. **Label Binarizer**: عشان نرجع من الأرقام للتصنيفات

---

## 1️⃣3️⃣ اختبار النموذج المحفوظ (Test Saved Model)

```python
from keras.models import load_model

# تحميل المكونات المحفوظة
loaded_model = load_model("subject_area_model.keras")

with open('tfidf_vectorizer.pkl', 'rb') as f:
    loaded_vectorizer = pickle.load(f)

with open('label_binarizer.pkl', 'rb') as f:
    loaded_mlb = pickle.load(f)

loaded_vocab = loaded_mlb.classes_
```

### دالة التنبؤ
```python
def predict_subject_areas(abstract_text, model, vectorizer, mlb, threshold=0.5):
    # تحويل النص إلى أرقام
    abstract_vector = vectorizer.transform([abstract_text])
    abstract_vector_dense = abstract_vector.toarray()
    
    # التنبؤ
    predictions = model.predict(abstract_vector_dense)
    
    # تحويل الاحتمالات إلى قرارات (أكبر من 0.5 = موجود)
    binary_predictions = (predictions[0] > threshold).astype(int)
    
    # تحويل الأرقام إلى تصنيفات
    predicted_subjects = [mlb.classes_[i] for i, val in enumerate(binary_predictions) if val == 1]
    
    return predicted_subjects
```

**كيف تشتغل؟**
1. تاخد النص
2. تحوله لأرقام بنفس طريقة التدريب
3. تدخله للنموذج
4. النموذج يطلع احتمالات (0 إلى 1)
5. أي احتمال أكبر من 0.5 يعتبر تصنيف موجود
6. ترجع التصنيفات النهائية

---

## 1️⃣4️⃣ مثال عملي

```python
input_abstract = """The dominant sequence transduction models are based on complex
recurrent or convolutional neural networks in an encoder-decoder configuration.
The best performing models also connect the encoder and decoder through an attention mechanism.
We propose a new simple network architecture, the Transformer, based solely on attention mechanisms,
dispensing with recurrence and convolutions entirely..."""

predicted_terms = predict_subject_areas(input_abstract, loaded_model, loaded_vectorizer, loaded_mlb)
print("Predicted subject areas:", predicted_terms)
```

**هذا المثال:**
- ملخص ورقة الـ Transformer الشهيرة
- النموذج هيتوقع إنها في مجالات زي: Machine Learning, Natural Language Processing, إلخ

---

## الخلاصة النهائية 🎯

### ما عمله الكود:
1. ✅ قرأ بيانات الأوراق البحثية
2. ✅ نظف البيانات وأزال التكرارات
3. ✅ قسم البيانات لتدريب وتحقق واختبار
4. ✅ حول النصوص لأرقام بـ TF-IDF
5. ✅ حول التصنيفات لـ binary encoding
6. ✅ بنى نموذج deep learning بـ 4 طبقات مخفية
7. ✅ درب النموذج مع منع overfitting
8. ✅ قيم الأداء ورسم النتائج
9. ✅ حفظ النموذج للاستخدام لاحقاً
10. ✅ اختبر النموذج على مثال حقيقي

### النتيجة:
نموذج ذكي يقدر يشوف ملخص أي ورقة بحثية ويقول إيه مجالها العلمي! 🚀