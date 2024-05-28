from sklearn.datasets import load_iris
from xgboost import XGBClassifier
import streamlit as st
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import numpy as np
data = load_iris()
X, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBClassifier(colsample_bytree=0.8, learning_rate=0.01, max_depth=4, n_estimators=50, subsample=0.8)
model.fit(x_train, y_train)
class_names = data.target_names

st.title('这是一个鸢尾花分类器应用')
st.write('请输入花的四个特征来预测花的种类')
sepal_length = st.number_input('sepal length (cm)', min_value=0.1,step=1.0)
sepal_width = st.number_input('sepal_width (cm)', min_value=0.1,step=1.0)
petal_length = st.number_input('petal_length (cm)', min_value=0.1,step=1.0)
petal_width = st.number_input('petal_width (cm)', min_value=0.1,step=1.0)
if sepal_length == 0 or sepal_width == 0 or petal_length == 0 or petal_width == 0:
    st.warning('所有输入值必须大于0！')
if st.button('predict'):
    input_features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(input_features)[0]
    predicted_class = class_names[prediction]
    st.write(f'它可能是{predicted_class}')
cv_scores = np.mean(cross_val_score(model,X, y, cv=10, scoring='accuracy'))
cv_scores=round(cv_scores,2)
st.write(f'模型的准确率是{cv_scores}')
