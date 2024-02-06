import joblib as jb 
import streamlit as st 


load_model=jb.load('model.py')
load_vector=jb.load('vector.py') 


def get_keys(val,my_dict): 
    for key, value in my_dict.items():
        if val==value:
            return key

def main():
     
      st.header('Classification of news') 
      news=st.text_area('Enter the news here') 
      model=['linear _svc']
      st.selectbox('select a model',model)
      prediction_labels={'fake_news':0, 'true_news':1}
      if st.button('classify'):
          st.text('original text ::\n{}'.format(news))
          vec_x=load_vector.transform([news])
          predictions=load_model.predict(vec_x) 
          result=get_keys(predictions,prediction_labels) 
          st.success(result)






if __name__ =='__main__' :
    main()