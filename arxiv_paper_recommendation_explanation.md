# شرح الكود 🚀

## 📚 القسم الأول: استيراد المكتبات

```python
import pandas as pd
```
**🔍 الشرح**: استيراد pandas للتعامل مع البيانات في جداول

```python
import numpy as np
```
**🔢 الشرح**: استيراد numpy للعمليات الرياضية

---

## 📖 القسم التاني: قراءة البيانات

```python
df = pd.read_csv("arxiv_data_210930-054931.csv")
```
**📥 الشرح**: قراءة ملف CSV فيه بيانات الأوراق العلمية

```python
df.head()
```
**👀 الشرح**: عرض أول 5 صفوف عشان نشوف شكل البيانات

```python
df.shape
```
**📏 الشرح**: معرفة عدد الصفوف والأعمدة (مثلاً: 1000 ورقة × 5 معلومات)

---

## 🧹 القسم التالت: تنظيف البيانات

```python
df.duplicated().sum()
```
**🔍 الشرح**: عد الصفوف المتكررة

```python
df.drop_duplicates(inplace=True)
```
**🗑️ الشرح**: مسح الصفوف المتكررة من البيانات الأصلية

```python
df.duplicated().sum()
```
**✅ الشرح**: التأكد إن المسح اشتغل (المفروض يطلع 0)

---

## 🔗 القسم الرابع: تحضير النصوص

```python
df["content"] = df["titles"] + " " + df["abstracts"]
```
**📝 الشرح**: دمج العنوان والملخص في عمود واحد للتحليل

```python
titles = df["titles"]
abstracts = df["abstracts"]
```
**💾 الشرح**: حفظ العناوين والملخصات في متغيرات منفصلة

```python
!pip install -U -q sentence-transformers
```
**📦 الشرح**: تنصيب مكتبة تحويل النصوص لأرقام

---

## 🤖 القسم الخامس: تحميل النموذج السحري

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
```
**🧠 الشرح**: استيراد النموذج وأداة حساب التشابه

```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```
**⚡ الشرح**: تحميل نموذج ذكي (80 ميجا، مدرب على مليار جملة)

```python
embeddings = model.encode(df["content"].tolist(), show_progress_bar=True)
```
**🎯 الشرح**: تحويل كل النصوص لمصفوفات أرقام (embeddings) - ده القلب النابض للنظام!

---

## 💾 القسم السادس: حفظ الملفات

```python
import pickle
```
**📦 الشرح**: استيراد أداة حفظ البيانات

```python
with open('embeddings.pkl', 'wb') as f:
    pickle.dump(embeddings, f)
```
**💿 الشرح**: حفظ الـ embeddings (الجزء الأهم!)

```python
with open('titles.pkl', 'wb') as f:
    pickle.dump(titles, f)
    
with open('abstracts.pkl', 'wb') as f:
    pickle.dump(abstracts, f)
    
with open('rec_model.pkl', 'wb') as f:
    pickle.dump(model, f)
```
**📁 الشرح**: حفظ العناوين والملخصات والنموذج

---

## 🔄 القسم السابع: تحميل الملفات المحفوظة

```python
embeddings = pickle.load(open('models/embeddings.pkl','rb'))
titles = pickle.load(open('models/titles.pkl','rb'))
abstracts = pickle.load(open('models/abstracts.pkl','rb'))
rec_model = pickle.load(open('models/rec_model.pkl','rb'))
```
**📂 الشرح**: تحميل كل الملفات المحفوظة من مجلد models

---

## 🎯 القسم الثامن: الدالة السحرية

```python
def recommend_papers(input_title, top_n=5):
```
**🔮 الشرح**: بداية دالة الترشيح (افتراضياً بترجع 5 أوراق)

```python
    input_embedding = rec_model.encode([input_title])
```
**⚡ الشرح**: تحويل العنوان المطلوب لـ embedding

```python
    sim_scores = cosine_similarity(input_embedding, embeddings)[0]
```
**🎲 الشرح**: حساب التشابه بين العنوان وكل الأوراق الموجودة (ده الجزء المهم!)

```python
    top_indices = sim_scores.argsort()[::-1][:top_n]
```
**🏆 الشرح**: ترتيب النتائج من الأعلى تشابه للأقل وأخذ أحسن النتائج

```python
    return df.iloc[top_indices][['titles', 'abstracts']]
```
**🎁 الشرح**: إرجاع العناوين والملخصات للأوراق الأشبه

---

## 🧪 القسم التاسع: تجربة النظام

```python
recommended = recommend_papers("Attention Is All You Need", top_n=5)
```
**🚀 الشرح**: تجربة النظام بالبحث عن أوراق شبيهة بورقة "Attention Is All You Need" الشهيرة

```python
recommended
```
**📊 الشرح**: عرض النتائج

---

## 🎉 الخلاصة السريعة
```
📖 قراءة البيانات → 🧹 تنظيفها → 🤖 تحويلها لأرقام → 💾 حفظها → 🔍 البحث والترشيح
```

**💡 الفكرة الأساسية**: النظام بيحول النصوص لأرقام عشان الكمبيوتر يفهمها، وبعدين بيحسب التشابه رياضياً بين النصوص ويرجعلك الأشبه! 🎯✨