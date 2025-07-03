import pandas as pd
import re
import fasttext
import time
from sklearn.model_selection import train_test_split

train_data_path = './Classification/content/train_data.txt'
test_data_path = './Classification/content/test_data.txt'
model_path = './Classification/models/fasttext_model_tnn.bin'
data_path = './Classification/content/dataset.csv'



# Danh sách stopwords tiếng Việt
stopwords = [
    "và", "hoặc", "có", "của", "cho", "là",',','.','bị', 'bởi', 'cả', 'các', 'cái', 'cần', 'càng', 'chỉ', 'chiếc', 'cho', 'chứ', 'chưa', 'chuyện', 'có', 'có_thể', 'cứ', 'của', 'cùng', 'cũng', 'đã', 'đang', 'đây', 'để', 'đến nỗi', 'đều', 'điều', 'do', 'đó', 'được', 'dưới', 'gì', 'hơn', 'ít', 'khi', 'không', 'là', 'lại', 'lên', 'lúc', 'mà', 'mỗi', 'một', 'một cách', 'này', 'nên', 'nếu', 'ngay', 'nhất', 'nhiều', 'như', 'nhưng', 'những', 'nơi', 'nữa', 'ở', 'phải', 'qua', 'ra', 'rằng', 'rằng', 'rất', 'rất', 'rồi', 'sau', 'sẽ', 'so', 'sự', 'tại', 'theo', 'thì', 'trên', 'trong', 'trước', 'từ', 'từng', 'và', 'vẫn', 'vào', 'vậy', 'về', 'vì', 'việc', 'với', 'vừa'
]

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

# Tải và tiền xử lý bộ dữ liệu
data = pd.read_csv(data_path, encoding="utf-8", sep=";", on_bad_lines="skip")

# Tiền xử lý dữ liệu
data.dropna(subset=['text'], inplace=True)  # Remove rows where 'text' is NaN
data['text'] = data['text'].apply(preprocess_text)

# Định dạng dữ liệu cho FastText
data['formatted'] = data['label'].astype(str) + ' ' + data['text']

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_data, test_data = train_test_split(data['formatted'], test_size=0.2, random_state=42)

train_data = train_data.str.strip()
test_data = test_data.str.strip()
# Lưu dữ liệu đã tiền xử lý vào các file


train_data.to_csv(train_data_path, index=False, header=False)
test_data.to_csv(test_data_path, index=False, header=False)

start_time = time.time()
# Huấn luyện mô hình

model = fasttext.train_supervised(input=train_data_path, epoch=100, lr=1.0, wordNgrams=3, verbose=2, minCount=1, dim=100)
print(f'Time: {time.time() - start_time} seconds')

# Lưu mô hình
model.save_model(model_path)



