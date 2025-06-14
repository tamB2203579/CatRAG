# Import các thư viện cần thiết
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import unicodedata
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,f1_score, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from underthesea import text_normalize, word_tokenize
from wordcloud import WordCloud

# Đọc file CSV chứa dữ liệu, phân tách bởi dấu chấm phẩy
df = pd.read_csv('./content/classification.csv', sep=';')

# Danh sách các từ dừng (stopwords) tiếng Việt sẽ được loại bỏ trong quá trình tiền xử lý
stopwords = [
    "ngành","điểm chuẩn","chỉ tiêu", "và", "hoặc", "có", "của", "cho", "là",',','.','bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'chuyện', 'có', 'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến nỗi', 'đều', 'điều', 'do', 'đó', 'được', 'dưới', 'gì', 'hơn', 'ít', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một', 'một cách', 'này', 'nên', 'nếu', 'ngay', 'nhất', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'ở', 'phải', 'qua', 'ra', 'rằng', 'rằng', 'rất', 'rất', 'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 'trong', 'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'về', 'vì', 'việc', 'với', 'vừa'
]

# Mapping các nhãn văn bản thành các số (label -> số nguyên)
category_mapping = {
    '__label__Dao_tao': 0,
    '__label__Hoc_tap_va_ren_luyen': 1,
    '__label__Khen_thuong_va_ky_luat': 2,
    '__label__Ky_tuc_xa': 3,
    '__label__Hoc_phi': 4,
    '__label__Nhung_muc_khac': 5,

}

# Hàm tiền xử lý văn bản
def preprocess_text(text):
    if not isinstance(text, str):
        return "" # Return an empty string for missing values
    text = text.lower()  # Chuyển về chữ thường
    text = re.sub(r'\d+', '', text)  # Loại bỏ chữ số
    text = re.sub(r'\s+', ' ', text)  # Loại bỏ khoảng trắng thừa
    text = re.sub(r'[^\w\s]', '', text)  # Loại bỏ dấu câu
    # Loại bỏ stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    return text

# Chuẩn hóa unicode theo chuẩn NFC
def unicode_normalize(text):
    return unicodedata.normalize('NFC', text)

# Gán nhãn số vào cột mới
df['Labels1'] = df['Label'].map(category_mapping)

# Vẽ biểu đồ phân bố nhãn
df['Labels1'].plot(kind='hist', bins=20, title='Label')
plt.gca().spines[['top', 'right',]].set_visible(False)

# Chuẩn hóa unicode và tiền xử lý câu hỏi
df['Question1'] = df['Question'].apply(unicode_normalize).apply(preprocess_text)

# Tạo bản sao của DataFrame để xử lý tokenize
tkn_df = df.copy()

# Tách từ bằng thư viện underthesea (dạng text có dấu gạch dưới giữa các từ ghép)
tkn_df['Question1'] = tkn_df['Question1'].apply(lambda sentence: word_tokenize(sentence, format='text'))


# Loại bỏ dấu gạch dưới trong các từ ghép để dễ đọc hơn
temp_tkn_df = tkn_df.copy()
temp_tkn_df['Question1'] = temp_tkn_df['Question1'].apply(lambda x: [word.replace('_', ' ') for word in x.split()])

# Lấy danh sách tất cả các từ
all_words = [word for title in temp_tkn_df['Question1'] for word in title]

# Đếm tần suất xuất hiện của từng từ
word_frequency = {}
total_titles = len(temp_tkn_df)

for word in all_words:
    if word in word_frequency:
        word_frequency[word] += 1
    else:
        word_frequency[word] = 1

# Chuẩn hóa tần suất theo số lượng câu hỏi (tần suất tương đối)
word_frequencies = {word: freq / total_titles for word, freq in word_frequency.items()}

# Chuyển dữ liệu tần suất thành DataFrame để hiển thị
word_frequency_df = pd.DataFrame(word_frequencies.items(), columns=['Word', 'Frequency'])
word_frequency_df = word_frequency_df.sort_values(by='Frequency', ascending=False)


word_frequency_df = word_frequency_df.reset_index(drop=True)


corpus = tkn_df.copy()
corpus['Question1'] = corpus['Question1'].astype(str)

vectorizer = TfidfVectorizer(min_df=10)
features = vectorizer.fit_transform(corpus['Question1'])

joblib.dump(vectorizer, 'tfidf.joblib')
tfidf_df = pd.DataFrame(data=features.toarray(), columns=vectorizer.get_feature_names_out())

tfidf_df['Labels1'] = corpus['Labels1']
print(tfidf_df.head(70))