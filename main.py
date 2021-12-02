from os import write
import streamlit as st
 

import numpy as np
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


header = st.beta_container()
body = st.beta_container()
classify_container = st.beta_container()



######################## Summarization code  ########################################


def classify(a):
    filename = 'model.sav'
    model_reloaded = pickle.load(open(filename, 'rb'))
    
    te =[]
    te.append(a)
    ab = model_reloaded.predict_proba(te)
    np.set_printoptions(formatter={'float_kind':'{:f}'.format})
    result = ab.tolist()
    test_res = result[0]
    li_goals = ["No Poverty","Zero Hunger","Good Healthand Well Being","Quality Education"
            ,"Gender Equality","Clean Water and Sanitation","Affordable  and Clean Energy"
            ,"Decent Work and Economic Growth","Industry,Innovation and Infrastructure"
            ,"Reduced Inequalites","Sustainable Cities and Communities",
            "Responsible Consumption and Production","Climate Action","Life Below Water","Life On Land"]
    t =zip(li_goals,test_res)
    df_predic = pd.DataFrame(t,columns=["SDG Category","Score"])
    df_predic.index = df_predic.index + 1
    fi= df_predic.sort_values("Score", ascending = [False])
    return((fi))
    


with header:
    titl, imga = st.beta_columns(2)
    titl.title('UNO SDG CLASSIFIER')
    imga.image('1sdg_logo.svg.gif')
    upload_col = st.beta_columns(1)
    upload_col.subheader()
    uploaded_file = upload_col.file_uploader(
        'Choose your .txt file', type="txt")
    if uploaded_file is not None:
        rawtext = str(uploaded_file.read(), 'utf-8')
    if st.button('Results here'):
        with classify_container:
            if rawtext == "":
                st.header('Classification :)')
                st.write('Please enter text or upload a file to see the Classification')
            else:
                result = classify(rawtext)
                st.header('SDG Classification :)')
                #res, plot = st.beta_columns(2)
                st.dataframe(result)
                df = pd.DataFrame(result, columns = ["Score"])
                st.bar_chart(df)

               
