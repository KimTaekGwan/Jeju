# 필요한 라이브러리를 임포트합니다.
import gdown
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
import os


import random
import pandas as pd
import numpy as np
import pandas as pd
import pickle as pkl
import joblib
import holidays

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
# from tensorflow.keras.callbacks import EarlyStopping

import yaml
with open("config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



base_path = './data/'
train_data = pd.read_csv(base_path + 'train.csv')

def pre_train_df():
    train_df = train_data.copy()

    train_df = train_df[['ID','price(원/kg)']]
    train_df.columns = ['ID','price']
    return train_df

all_data = pre_train_df()

def split_id(x):
    return x.split('_')

def extract_id(x):
    return x[0]

def pre_init_feature(df):
    """ID에서 품목, 유통법인코드, 지역코드 추출, 날짜 추출"""
    print('pre_init_feature')
    split_series = df['ID'].apply(split_id)
    df['item'] = split_series.apply(lambda x: x[0])
    df['corporation'] = split_series.apply(lambda x: x[1])
    df['location'] = split_series.apply(lambda x: x[2])
    df['ymd'] = split_series.apply(lambda x: x[3])
    
def pre_ymd(df):
    print('pre_ymd')
    # year, month, day 생성
    df['year'] = df['ymd'].apply(lambda x : int(x[:4]))
    df['month'] = df['ymd'].apply(lambda x : int(x[4:6]))
    df['day'] = df['ymd'].apply(lambda x : int(x[6:]))
    
def pre_timestamp(df):
    """year, month, day 칼럼을 사용하여 datetime 형식인 ts 칼럼 생성"""
    print('pre_timestamp')
    df['ts'] = df.apply(lambda x : pd.Timestamp(year=x.year,
                                                month=x.month,
                                                day=x.day),axis=1)
    
def pre_weekday(df):
    """ts 칼럼을 사용하여 weekday 칼럼 생성"""
    print('pre_weekday')
    df['weekday'] = df['ts'].dt.weekday
    
def pre_holiday(df):
    """ts 칼럼을 사용하여 holiday 칼럼 생성"""
    print('pre_holiday')
    kr_holidays = holidays.KR()
    df['holiday'] = df['weekday'].apply(lambda x : 1 if x == 6 else 0)
    df['holiday'] = df.apply(lambda x : 1 if x.ts in kr_holidays
                                        else x.holiday, axis=1)

def pre_drop(df):
    """불필요한 칼럼 삭제"""
    df.drop(['ID', 'ymd'], axis=1, inplace=True)
    
# 카테고리형 변수 리스트 설정
cat_cols = ['item', 'corporation', 'location',
            'year', 'month', 'weekday', 'holiday']

def pre_cat_label_encoder(df):
    """카테고리형 변수들을 LabelEncoder로 변환 후, 각 인코더를 딕셔너리에 저장"""
    # 카테고리형 변수 리스트 설정
    cat_cols = ['item', 'corporation', 'location',
                'year', 'month', 'weekday', 'holiday']
    label_encoders = {}
    for col in cat_cols:
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
        
    # pickle로 저장
    pkl.dump(label_encoders, open('label_encoders.pkl', 'wb'))
    return label_encoders


def pre_cat_label_encoder_test(df):
    """test 데이터 변환 시 사용"""
    # pickle로 저장한 label_encoders를 불러와서 정의하기
    label_encoders = pkl.load(open('label_encoders.pkl', 'rb'))
    
    for col in cat_cols:
        try:
            df[col] = label_encoders[col].transform(df[col])
        except:
            print('train에 없는 값이 test에 존재합니다.')
            df[col] = 0
            
def pre_cat_onehot_encoder(df):
    """카테고리형 변수들을 OneHotEncoder로 변환 후 N-1 인코딩을 적용"""
    onehot_encoders = {}
    for col in cat_cols:
        onehot_encoders[col] = OneHotEncoder(sparse=False, handle_unknown='ignore')
        encoded_data = onehot_encoders[col].fit_transform(df[[col]])
        
        # 원-핫 인코딩된 데이터를 DataFrame으로 변환
        encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoders[col].get_feature_names_out([col]))
        
        # 첫 번째 열(기준 범주)을 제거하여 N-1 인코딩 구현
        encoded_df = encoded_df.iloc[:, 1:]
        
        # 원래 DataFrame에 병합하고, 원본 열 삭제
        df = df.drop(col, axis=1).join(encoded_df)
        
    # pickle로 저장
    pkl.dump(onehot_encoders, open('onehot_encoders.pkl', 'wb'))
    return df, onehot_encoders

def pre_cat_onehot_encoder_test(df):
    """테스트 데이터 변환 시 사용"""
    onehot_encoders = pkl.load(open('onehot_encoders.pkl', 'rb'))
    print(onehot_encoders)
    
    for col in cat_cols:
        try:
            encoded_data = onehot_encoders[col].transform(df[[col]])
            encoded_df = pd.DataFrame(encoded_data, columns=onehot_encoders[col].get_feature_names_out([col]))
            
            # 첫 번째 열(기준 범주)을 제거하여 N-1 인코딩 구현
            encoded_df = encoded_df.iloc[:, 1:]
            
            df = df.drop(col, axis=1).join(encoded_df)
        except:
            print(f'{col} 열에 train에 없는 값이 test에 존재합니다.')
            
    return df

def pre_all_onehot(df):
    """전체 카테고리형 변수 전처리 과정"""
    pre_init_feature(df)
    pre_ymd(df)
    pre_timestamp(df)
    pre_weekday(df)
    pre_holiday(df)
    pre_drop(df)
    
    
    df = pre_cat_onehot_encoder_test(df)
    print(df)
    return df

# 분류 모델과 회귀 모델을 합친 모델 생성
class TotalModel():
    def __init__(self, clf_model, reg_model):
        self.clf_model = clf_model
        self.reg_model = reg_model
        
    def predict(self, x):
        clf_pred = self.clf_model.predict(x)
        clf_pred = np.where(clf_pred >= 0.5, 1, 0)
        reg_pred = self.reg_model.predict(x)
        
        # 분류 모델의 예측이 1이면 회귀 모델의 예측값을 반환
        # 분류 모델의 예측이 0이면 0을 반환
        pred = [reg_pred[i] if clf_pred[i] == 1 else 0 for i in range(len(clf_pred))]
        history_dict = {'분류 모델 결과': clf_pred,
                        '예측 모델 결과': reg_pred,
                        'pred': pred}
        return pred, history_dict
    
# 구글 드라이브를 이용한 모델 다운로드
def download_model_file(url):
    output = "reg_model.pkl"
    gdown.download(url, output, quiet=False)

@st.cache_resource
def reg_load_model():
    '''
    Return:
        model: 구글 드라이브에서 가져온 모델 return 
    '''
    
    if not os.path.exists("reg_model.pkl"):
        print("Downloading model...")  
        download_model_file(config['model_path'])
    
    model_path = 'reg_model.pkl'
    model = joblib.load(model_path)

    return model

    
def load_model():
    clf_model = tf.keras.models.load_model('./model/model_basic.h5')
    reg_model = reg_load_model()
    total_model = TotalModel(clf_model, reg_model)
    return total_model

total_model = load_model()



items_dict = {'TG': '감귤', 'CR': '당근', 'CB': '양배추',
              'RD': '무', 'BC': '브로콜리'}
corporations_dict = {'A': 'A회사', 'B': 'B회사', 'C': 'C회사',
                     'D': 'D회사', 'E': 'E회사', 'F': 'F회사'}
locations_dict = {'J': '제주도 제주시', 'S': '제주도 서귀포시'}



def pre_test_df(ID):
    test_df = pd.DataFrame({'ID': [ID],
                            'price': [0]})
    return test_df

def run_model(item, corporation, location, d):
    id = '_'.join([item, corporation, location, d.strftime("%Y%m%d")])
    # st.write("ID: ", id)
    
    test_df = pre_test_df(id)
    
    # 전처리
    test_df_1 = pre_all_onehot(test_df)
    
    
    test_df_1 = test_df_1.drop(columns=['ts'])
    X_test = test_df_1.drop('price', axis=1)
    
    # 모델 불러오기
    pred, historys = total_model.predict(X_test)
    
    history_dict = {'id': id,
                    'Feature Engineering': test_df,
                    'One-hot Encoding': test_df_1}
    
    for key, value in historys.items():
        history_dict[key] = value
    
    return pred[0], history_dict

def search_price(item, corporation, location, d):
    id = '_'.join([item, corporation, location, d.strftime("%Y%m%d")])
    
    # all_data에서 id에 해당하는 행 추출
    df = all_data[all_data['ID'] == id]
    
    # if df가 비어있지 않으면
    if not df.empty:
        # price 칼럼 반환
        return True, df['price'].values[0]
    else:
        # 비어있으면 0 반환
        return False, '해당 실제 데이터가 없습니다.'

# Streamlit 애플리케이션을 정의합니다.
def run_app():
    # 애플리케이션 제목을 설정합니다.
    st.title('제주 특산물 가격 예측 AI')

    # 파일 업로드 위젯을 생성합니다.
    with st.form("my_form"):
        st.write("ID 입력")
        item = st.selectbox('Item', ['TG', 'CR', 'CB', 'RD', 'BC'])
        corporation = st.selectbox('Corporation', ['A', 'B', 'C', 'D', 'E', 'F'])
        location = st.selectbox('Location', ['J', 'S'])
        d = st.date_input("Date", datetime.date(2022, 7, 6))
        
        submit = st.form_submit_button('예측하기')
        
    if submit:
        pred, history_dict = run_model(item, corporation, location, d)
        search, actual_p = search_price(item, corporation, location, d)
        
        col1, col2 = st.columns(2)
        
        # col1에는 예측값과 실제값 결과와, 해당 결과에 대한 설명(예측값과 실제값의 차이)
        col1.write(f"예측값: {pred}")
        
        # col2에는 실제값과 예측값 bar plot 시각화
        if search:
            col1.write(f"실제값: {actual_p}")
            col1.write(f"차이: {abs(pred - actual_p)}")
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(['Actual', 'Predict'], [actual_p, pred])
            ax.set_title('Actual vs. Predict')
            ax.set_ylabel('Price')
            plt.xticks(rotation=0)
            col2.pyplot(fig)
        else:
            col2.write(actual_p)
            
        # 예측 과정 expander에 넣기
        with st.expander("예측 과정"):
            for key, value in history_dict.items():
                st.subheader(key)
                st.write(value)
                st.write('---')
        
    
        

# Streamlit 애플리케이션을 실행합니다.
if __name__ == "__main__":
    run_app()
