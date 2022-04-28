from scipy.sparse import data
import streamlit as st
import pandas as pd
import streamlit as st
import itertools
import pandas as pd 
from sklearn.model_selection import train_test_split
import numpy as np 
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt 
import plotly.express as px
import matplotlib
import seaborn as sns
from sklearn.pipeline import Pipeline
# from wordcloud import WordCloud
# from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from scipy.stats import chi2_contingency,chi2
# import statsmodels.api as sm 
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
from scipy.stats import anderson
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix 
from PIL import Image
import sweetviz as sv
import codecs

from sklearn import preprocessing, model_selection, metrics, feature_selection
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.svm import SVC
# from sklearn.metrics import confusion_matrix
from sklearn import neighbors, linear_model, svm, tree, ensemble
# from wordcloud import WordCloud, STOPWORDS
from sklearn.ensemble import AdaBoostClassifier
# from sklearn.decomposition import PCA
# from IPython.display import display, HTML
import plotly.graph_objs as go
# from plotly.offline import init_notebook_mode,iplot

from pandas_profiling import ProfileReport 
import streamlit.components.v1 as components
from streamlit_pandas_profiling import st_profile_report
matplotlib.use("Agg")

import sklearn.metrics as metrics
import seaborn as sns
# import altair as alt
# import pydeck as pdk
import base64
from matplotlib.pyplot import figure
import squarify
from pandas import DataFrame
###########################################################
import warnings 
warnings.filterwarnings('ignore')
import datetime as dt
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from math import isnan
from sklearn.model_selection import GridSearchCV
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import GridSearchCV
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
#######################################
from sklearn.metrics import accuracy_score
###for association
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.neighbors import KNeighborsClassifier
#############
import keras
# import tensorflow as tf
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import r2_score as r2s
from math import sqrt
import os

import datetime
from datetime import datetime
st.set_page_config(
     page_title="CRM-TMA",
     page_icon="🧊",
     layout="wide", 
     initial_sidebar_state="expanded")
# new_title = '<p style="font-family:serif; font-weight:bold;color:Green;font-size: 60px;">CRM-TMA</p>'
# st.markdown(new_title, unsafe_allow_html=True)
new_title = '<p style="font-family:serif; font-weight:bold;color:Green;font-size: 60px;">ग्राहक-360</p>'

# imgze = Image.open("./images/title.PNG")
# st.sidebar.image(imgze,use_column_width='auto')


file_8 = open("./images/video (2).gif", "rb")
contents88 = file_8.read()
data_url88 = base64.b64encode(contents88).decode("utf-8")
file_8.close()
st.sidebar.markdown(f'<img src="data:image/gif;base64,{data_url88}" alt="cat gif" width="300" height="220">',unsafe_allow_html=True,)


colT1,colT2, colT3 = st.columns(3)
with colT2:
#      st.title('CRM-TMA')
#      st.markdown(new_title, unsafe_allow_html=True)
       imgq = Image.open("./images/title.PNG")
       st.image(imgq,use_column_width='auto')
import streamlit.components.v1 as components

col6,col3, col4, col5 = st.columns([0.5,1.5,0.5,3])
with col3:
  imgt = Image.open("./images/grahak-360.PNG")
  st.image(imgt,use_column_width='auto')
with col5:
  j = '''Attention decision-makers!! Are you tired of skimming through oodles
  of spreadsheets? Looking for a comprehensive contrived CRM tool to alleviate
  your prudent judgments? Then, by landing here, you have hit the bullseye!
  This Grahak360 built up, with all-inclusive features, will aid you with
  predictive analytics. It will guide you to get a panorama of your data.
  The tool will assist you to segment and classify your customers.
  It will help you create association rules for product recommendations
  and support you to predict future transactions and your company’s net profit
  contribution to an overall future relationship with your customers.
  Also, it will benefit you to get an upper hand on forecasting future sales.
  In short, it will be your helping hand to do forensic analyses by analyzing
  the retention analytics of your customers. Do you want to get down to the 
  nitty-gritty and see it for yourself? Then grab your data and
  let's get started!!!'''
  st.write(j)

add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
st.markdown(add_line, unsafe_allow_html=True)

st.sidebar.title('USE SIDEBAR TO EXECUTE ACTIVITIES')
st.sidebar.markdown('Let’s start with GRAHAK 360 !!')
#Taking File input
# global dff 
global mm
global RFM_table

col1, col2 = st.columns(2)
with col1:
  data_file = st.file_uploader("Upload Main CSV",type=["csv"],  key = '89')
  if data_file is not None:
    file_details = {"filename":data_file.name,"filetype":data_file.type,"filesize":data_file.size}
    st.write(file_details)
    dff = pd.read_csv(data_file)
    dff['TotalAmount'] = dff.apply(lambda row: (row['Quota']*row['Amount']),axis=1)
    # dff['CustomerID'] = dff['CustomerID'].astype(np.int64)
    dff['BillDate'] = pd.to_datetime(dff['BillDate'])
    dff.dropna(axis = 0, subset = ['Product', 'CustomerID'], inplace = True)
    if st.sidebar.checkbox('Display Main Data',False, key = '29'):
      st.subheader('Show Main Input dataset')
      st.write(dff)
      st.write('''Take a look at the Main dataset you have fed to our CRM tool. Check for the attributes
and data values before moving further!!''')
    count = 1
    if st.sidebar.checkbox('Access Input Main Data ', key = '4254354'):
      CI1= st.number_input('Enter a Customer ID',0, 9000000, 0, 1, key = '1878')
      for i in range(len(dff)):
          if dff.CustomerID[i] == CI1:
               st.subheader('Transaction No. {}'.format(count))
               st.write('The Bill id of this Customer is {},This Customer Belongs to {}.The Product Purchased is {},its Merchnadise id is {} and the amount of Product Purchased is/are{} and the Price of each is {}.Total Transaction Amount is {}'.format(dff.Bill[i],dff.Country[i],dff.Product[i],dff.MerchandiseID[i],dff.Quota[i], dff.Amount[i],dff.TotalAmount[i]))
#                st.write('Bil :', dff.Bill[i])
#                st.write('country :',dff.Country[i])
#                st.write('Merchandise ID', dff.MerchandiseID[i])
#                st.write('Product :', dff.Product[i])
#                st.write('Quota: ', dff.Quota[i])
#                st.write('Amount: ', dff.Amount[i])
#                st.write('Total Amount', dff.TotalAmount[i])
               count = 1 + count
      st.write('Looks like this customer has a greater place in your revenue stream. Please find the details of this customer as per the dataset you provided.')
with col2:
  mm = st.file_uploader("Upload CSV for Prediction",type=["csv"], key = '8998')
  if mm is not None:
    file_details1 = {"filename":mm.name,"filetype":mm.type,"filesize":mm.size}
    st.write(file_details1)
    dff1 = pd.read_csv(mm)
    dff1['TotalAmount'] = dff1.apply(lambda row: (row['Quota']*row['Amount']),axis=1)
    dff1['BillDate'] = pd.to_datetime(dff1['BillDate'])
    # dff['CustomerID'] = dff['CustomerID'].astype(np.int64)
    dff1.dropna(axis = 0, subset = ['Product', 'CustomerID'], inplace = True)
    if st.sidebar.checkbox('Display Data of for prediction',False, key = '200'):
      st.subheader('Show Input dataset for prediction' )
      st.write(dff1)
      st.write('''Take a look at the dataset for prediction you have fed to our CRM tool. Check for the attributes 
and data values before moving further!!''')
    count1 = 1
    if st.sidebar.checkbox('Access Input Pred Data ',key = '698955312'):
      CI2= st.number_input('Enter a Customer ID',0, 9000000, 0, 1, key = '1875ss58')
      for i in range(len(dff1)):
          if dff1.CustomerID[i] == CI2:
               st.subheader('Transaction No.'.format(count1))
               st.write('The Bill id of this Customer is {}, This Customer Belongs to {}. The Product Purchased is {}, its Merchnadise id is {} and the amount of Product Purchased is/are {} and the Price of each is {}. Total Transaction Amount is {}'.format(dff1.Bill[i],dff1.Country[i],dff1.Product[i],dff1.MerchandiseID[i],dff1.Quota[i], dff1.Amount[i],dff1.TotalAmount[i]))
#                st.subheader('Transaction No.', count)
#                st.write('Bil :', dff1.Bill[i])
#                st.write('country :',dff1.Country[i])
#                st.write('Merchandise ID', dff1.MerchandiseID[i])
#                st.write('Product :', dff1.Product[i])
#                st.write('Quota: ', dff1.Quota[i])
#                st.write('Amount: ', dff1.Amount[i])
#                st.write('Total Amount', dff1.TotalAmount[i])
               count1 = count1 + 1
      st.write('Looks like this customer has a greater place in your revenue stream. Please find the details of this customer as per the dataset you provided.')

add_line1= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
st.markdown(add_line1, unsafe_allow_html=True)





class ghar():
     
     def intro(self):
          col7, col8 ,col9 = st.columns(3)
          with col7:
               img3 = Image.open("./images/Descriptive Analysis.png")
               st.subheader('1. Descriptive Analysis')
               st.image(img3,caption = 'Descriptive Analysis',use_column_width= None )
               st.write('''Get a perspective of your data. Help 
understand hidden patterns with the
sweetviz analysis, pandasprofiling 
and get a picture on the demographics
of your customers''')
          
          with col8:
               img4 = Image.open("./images/customer Segmentation.png")
               st.subheader('2. Customer Segmentation')
               st.image(img4,caption = 'Customer Segmentation',use_column_width= None )
               st.write('''Do smart segmentation to understand
your customers with our RFM and 
Hybrid segmentation techniques. Get 
real-time plots and access your 
customer with a click.''')
          with col9:
               img5 = Image.open("./images/customer Classification.png")
               st.subheader('3. Customer Classification')
               st.image(img5,caption = 'Customer Classification',use_column_width= None )
               st.write('''Helps you in making judicious decisions
by choosing the right classifier 
for the prediction of your 
customers.''')
          col10, col11 ,col12 = st.columns(3)
          with col10:
               img6 = Image.open("./images/Sale Forecasting.png")
               st.subheader('4. Sales Forecasting')
               st.image(img6,caption = 'Sales Forecasting',use_column_width= None )
               st.write('''Forecast your company’s future sales to 
get deeper insights for your next
plans of action.''')
          with col11:
               img17 =Image.open("./images/Product recommendation.png")
               st.subheader('5.Product Recommendation')
               st.image(img17,caption = 'Product Recommendation',use_column_width= None )
               st.write('''Create real-time association rules for 
product recommendation using 
market-basket analysis''')                
          with col12:
               img8 =Image.open("./images/Customer Retension.png")
               st.subheader('6. Forensic Analysis')
               st.image(img8,caption = 'Forensic Analysis',use_column_width= None )
               st.write('''Perform churn rate, cohort, and 
retention analysis. Get deeper insights
on monthly revenue, growth rate, active 
customers, total order, customer ratio 
and customer retention.''')  
		
          col13, col14 ,col15 = st.columns(3)
          with col14:
               img9 =Image.open("./images/CLTV.png")
               st.subheader('7. Customer Lifetime Value')
               st.image(img9,caption = 'Customer Lifetime Value',use_column_width= None )
               st.write('''Predict the expected future transactions
of your customer. Gauge your company's net
profit contribution to an overall future 
relationship with customers.''')                       
                       
               
             
               
          Vid1= open("./Add1.mp4", 'rb')
          Vid1_bytes = Vid1.read()
          st.video(Vid1_bytes)


class DataFrame_Loader():
  def __init__(self):
    print('Loading DataFrame')
  # def read_csv(self,data):
  #   self.df = pd.read_csv(data)
  #   return self.df
################################################################################
class EDA_Analysis():

  def __init__(self):
    print('General_EDA object created')

  def st_display_sweetiz(report_html,width = 1000, height = 500 ):
    report_file = codecs.open(report_html,'r')
    page = report_file.read()
    components.html(page,width=width,height=height, scrolling=True)
    HtmlFile = open("test.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code)
  
  
  def SweetV(self,x):
    st.subheader('SweetVIZ Data Analysis Report')
    analysis = sv.analyze([x,'EDA'])
    #analysis.show_html()
    analysis.show_html(filepath='./SWEETVIZ_REPORT.html', open_browser=False, layout='vertical',scale=1.0)
    # components.iframe(src='http://localhost:8501/EDA.html', width=1100, height=1200, scrolling=True)


    HtmlFile = open("SWEETVIZ_REPORT.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    print(source_code)
    components.html(source_code, height = 2000)
  def Exp1(self):
      with st.expander('See About SweetViz'):
          st.write(''' Sweetviz is a wonderful and very useful Python library that provides us with the EDA of a given dataset. Sweetviz let us perform a list of different analyses
          Single Dataset Analysis , Target Variable Analysis , Compare two datasets, Divide Dataset using boolean variable and Compare them.''')
      
   
  def Map(self):
    dict = {'lat':[], 'lon':[]}
    latlong = pd.DataFrame(dict)
    # coords = np.random.randn(1000, 2) / [50, 50] + [37.76, -122.4]
    # df = pd.DataFrame(coords, columns=["lat", "lon"])
    # numpy.random.randn(1) / [50, 50] + [37.76, -122.4]
    for i in range(0,len(dff)):
      if dff.Country[i]== 'United Kingdom':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [55.9, -4.75]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [55.9, -4.75] 
        

      elif dff.Country[i] == 'Germany':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] +[49.982, 8.27]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] +[49.982, 8.27]
        
      elif dff.Country[i] == 'France':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [45.89,6.116]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [45.89,6.116]
        
      elif dff.Country[i] == 'Sweden':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] +  [60.613, 15.60]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] +  [60.613, 15.60]
        
      elif dff.Country[i] == 'Finland':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [60.996, 24.4999]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [60.996, 24.4999]
        
      elif dff.Country[i] == 'EIRE':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [53.633, -8.183]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [53.633, -8.183]
        
      elif dff.Country[i] == 'Switzerland':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [47.3697, 7.349]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [47.3697, 7.349]
        
      elif dff.Country[i] == 'Greece':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [38.8989, 22.43458]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [38.8989, 22.43458]
        
      elif dff.Country[i] == 'Denmark':
        latlong.loc[len(latlong)]=np.random.randn(1) / [50, 50] +  [55.7090, 9.53449]
        latlong.loc[len(latlong)]=np.random.randn(2) / [50, 50] +  [55.7090, 9.53449]
        
      elif dff.Country[i] == 'Malta':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [35.937, 14.375]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [35.937, 14.375]
        
      elif dff.Country[i] == 'Australia':
        latlong.loc[len(latlong)]= np.random.randn(1) / [50, 50] + [-33.420, 151.3004]
        latlong.loc[len(latlong)]= np.random.randn(2) / [50, 50] + [-33.420, 151.3004]
        
      elif dff.Country[i] == 'Spain':
        latlong.loc[len(latlong)]= np.random.randn(1) / [50, 50] + [38.912, 6.337]
        latlong.loc[len(latlong)]= np.random.randn(2) / [50, 50] + [38.912, 6.337]
        
      elif dff.Country[i] == 'Belgium':
        latlong.loc[len(latlong)]= np.random.randn(1) / [50, 50] + [50.445, 3.9390]
        latlong.loc[len(latlong)]= np.random.randn(2) / [50, 50] + [50.445, 3.9390]
        
      elif dff.Country[i] == 'Netherlands':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [53.00, 6.5500]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [53.00, 6.5500]
        
      elif dff.Country[i] == 'Bahrain':
        latlong.loc[len(latlong)] =np.random.randn(1) / [50, 50] +  [26.066, 50.557]
        latlong.loc[len(latlong)] =np.random.randn(2) / [50, 50] +  [26.066, 50.557]
        
      elif dff.Country[i] == 'Denmark':
        latlong.loc[len(latlong)] = np.random.randn(1) / [50, 50] + [55.709, 9.5344]
        latlong.loc[len(latlong)] = np.random.randn(2) / [50, 50] + [55.709, 9.5344]
        
      elif dff.Country[i] == 'Portugal':
        latlong.loc[len(latlong)]= np.random.randn(1) / [50, 50] +[40.641, -8.657]
        latlong.loc[len(latlong)]= np.random.randn(2) / [50, 50] +[40.641, -8.657]
        
      else :
        latlong.loc[len(latlong)]= np.random.randn(1) / [50, 50] +[15.491, 73.815]
        latlong.loc[len(latlong)]= np.random.randn(2) / [50, 50] +[15.491, 73.815]
        

    st.map(latlong)
    add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
#########################################################################
class RFM_Analysis():
  def RFMvalues(self):
    
    dff['Bill'] = dff['Bill'].astype(np.int64)
    dff['BillDate'] = pd.to_datetime(dff['BillDate'])
    dff['CustomerID'] = dff['CustomerID'].astype(int)
    Last_Date = dff['BillDate'].max()
    RFM_table = dff.groupby(by = ['CustomerID'], as_index = False).agg({'BillDate':lambda x: (Last_Date- x.max()).days,'Bill': lambda x: len(x), 'TotalAmount':lambda x: x.sum()})
    RFM_table.rename(columns = {'BillDate':'Recency', 'Bill':'Frequency', 'TotalAmount':'Monetary'}, inplace = True)

    RFM_table["Recency_Score"] = pd.qcut(RFM_table['Recency'], 5, labels=[5, 4, 3, 2, 1])
    RFM_table["Frequency_Score"] = pd.qcut(RFM_table["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    RFM_table["Monetary_Score"] = pd.qcut(RFM_table["Monetary"], 5, labels=[1, 2, 3, 4, 5])
    RFM_table["RFM_Score"] = (RFM_table["Recency_Score"].astype(str) + RFM_table["Frequency_Score"].astype(str))

    seg_map = {r'[1-2][1-2]': 'Hibernating',r'[1-2][3-4]': 'At_Risk',r'[1-2]5': 'Cant_loose',r'3[1-2]': 'About_to_sleep',r'33': 'Need_attention',r'[3-4][4-5]': 'Loyal_customers',r'41': 'Promising',r'51': 'New_customers',r'[4-5][2-3]': 'Potential_loyalists',r'5[4-5]': 'Champions'}
    RFM_table['Segment'] = RFM_table['RFM_Score'].replace(seg_map, regex=True)    
    return RFM_table
  def Access_Cust(self):
    CI= st.number_input('Enter a Customer ID',0, 9000000, 0, 1, key = '1')
    for i in range(len(rfm.RFMvalues())):
      if rfm.RFMvalues().CustomerID[i] == CI:
        st.write('SEGMENT   :', rfm.RFMvalues().Segment[i])
        st.write('RFM SCORE :',rfm.RFMvalues().RFM_Score[i])

  def Scatter(self):
    numeric_columns = rfm.RFMvalues().columns.drop('CustomerID')
    st.sidebar.subheader("Scatter plot setup")
    # add select widget
    select_box1 = st.sidebar.radio(label='X axis', options=numeric_columns, key = '2')
    select_box2 = st.sidebar.radio(label="Y axis", options=numeric_columns, key = '3')
    fig = plt.figure(figsize=(6,2))
    plt.scatter(x=select_box1, y=select_box2, data=rfm.RFMvalues(),  color = 'blue',marker = '*', alpha = 0.3)
    plt.xlabel(select_box1)
    plt.ylabel(select_box2)
    st.pyplot(fig)

  def bar_plot(self):
    RFM_Segments= rfm.RFMvalues()["Segment"].value_counts()
    x = plt.figure(figsize=(6,2))
    sns.barplot(x=RFM_Segments.index,y=RFM_Segments.values)
    plt.xticks(rotation=45)
    plt.title('Customer Segments',color = 'black',fontsize=3)
    st.pyplot(x)
    # st.plotly_chart(x, use_container_width=True)
  def Tree_map(self):
    rd_treemap = rfm.RFMvalues().groupby('Segment').agg('count').reset_index()
    fig, ax = plt.subplots(1, figsize = (7,7)) 
    squarify.plot(sizes=rd_treemap['RFM_Score'], label=rd_treemap['Segment'], alpha=.8,color=['tab:red', 'tab:purple', 'tab:brown', 'tab:grey', 'tab:green'])
    plt.axis('off')
    st.pyplot(fig)
  
################################################################################
class Hybrid_Analysis():
  def Pre_hybrid(self):
    #Scaling the feature
    scale = StandardScaler()
    cols = ['Recency','Frequency','Monetary']
    RFM_table_scaler = scale.fit_transform(rfm.RFMvalues()[cols])
    #Linkage
    qq = plt.figure(figsize=(20,20))
    mergings = linkage(RFM_table_scaler,method = 'complete', metric = 'euclidean')
    dendrogram(mergings)
    plt.xlabel('Observations')
    plt.ylabel('Number of similarities')
    st.pyplot(qq)
  def skew_data(self):
    return rfm.RFMvalues().agg(['skew', 'kurtosis']).transpose()
  def Transform(self):
    RFM_h = rfm.RFMvalues().filter(['CustomerID', 'Recency', 'Frequency', 'Monetary'])
    RFM_h['r_quartile'] = pd.qcut(RFM_h['Recency'],5,labels  =  [5,4,3,2,1])
    RFM_h['f_quartile'] = pd.qcut(RFM_h['Frequency'].rank(method="first"), 5,labels =  [1,2,3,4,5])
    RFM_h['rev_quartile'] = pd.qcut(RFM_h['Monetary'],5,labels = [1,2,3,4,5])
    RFM_h['Monetary_T'] = np.log10(RFM_h['Monetary'])
    RFM_h['Frequency_T'] = np.cbrt(RFM_h['Frequency'])
    RFM_h['Frequency_T'] = np.cbrt(RFM_h['Frequency_T'])
    RFM_h['Frequency_T'] = np.sqrt(RFM_h['Frequency_T'])
    RFM_h['Recency_T'] = np.cbrt(RFM_h['Recency'])
    return RFM_h
  def Trans_data(self):
    COLUMNS = ['Recency','Frequency','Monetary','Recency_T','Monetary_T','Frequency_T']
    COL = st.sidebar.selectbox("Select Feature",COLUMNS, key = '7')
    ww = plt.figure(figsize=(6,2))
    sns.distplot(h_rfm.Transform()[COL])
    st.pyplot(ww)
  def rr(self):
    scaler = StandardScaler()
    scaler.fit(h_rfm.Transform())
    RFM_Table_scaled = scaler.transform(h_rfm.Transform())
    RFM_Table_scaled = pd.DataFrame(RFM_Table_scaled, columns = h_rfm.Transform().columns)
    XX = RFM_Table_scaled.iloc[:,4:7].values
    XX = pd.DataFrame(XX, columns = ['Monetary_T','Frequency_T', 'Recency_T'])
    return XX
  def elbow(self):
    Within_Cluster_Sum_of_Square = []
    for i in range(1,10,1):
      km1 = KMeans(n_clusters= i , init = 'k-means++',random_state = 50)
      km1.fit(h_rfm.rr())
      Within_Cluster_Sum_of_Square.append(km1.inertia_)
    zz  =plt.figure(figsize = (15,15))
    sns.set()
    plt.plot( range(1,10,1),(Within_Cluster_Sum_of_Square))
    plt.title('RFM Clustering: The Elbow Method')
    plt.ylabel('WCSS')
    plt.xlabel('Number of Clusters')
    st.pyplot(zz)

  def Best_K(self):
    r = []
    sil= []
    for y in range(2,10,1):
       km1 = KMeans(n_clusters= y, init= 'k-means++', random_state = 50)
       YY = km1.fit_predict(h_rfm.rr())
       XXX = h_rfm.rr()
       XXX['Clusters']  = km1.labels_  
       XX = h_rfm.Transform()
       XX = XX.astype('object')
       XX['Clusters'] = km1.labels_
       score = silhouette_score(XX, km1.labels_, metric='euclidean')
       sil.append(score)
    return sil.index(max(sil)) + 2

  def KM(self):
    # count = '0'
    # k  = st.slider('Enter a Expected number of Clusters ',min_value = 1,max_value = 10,  key = count)
    # count  = count + 'yy'
    # k = st.number_input("Enter value for K", min_value = 1,max_value = 10, value = 1, step = 1 , key  = '27')
    km1 = KMeans(n_clusters= h_rfm.Best_K(), init= 'k-means++', random_state = 50)
    YY = km1.fit_predict(h_rfm.rr())
    XXX = h_rfm.rr()
    XXX['Clusters']  = km1.labels_  
    XX = h_rfm.Transform()
    XX = XX.astype('object')
    XX['Clusters'] = km1.labels_
    return XX
  def BOX(self):
    # Opt = ['Recency','Frequency','Monetary','Monetary_T','Frequency_T','Recency_T']
    st.sidebar.subheader("box plot")
    # add select widget
    N = st.sidebar.radio(label='SELECT FEATURE of Box Plot', options=('Recency','Frequency','Monetary','Monetary_T','Frequency_T','Recency_T'), key = '2f6gd5d45gd5')
    # N = st.sidebar.selectbox("Select Feature",Opt, key = '5')
    ee = plt.figure(figsize=(6,2))
    sns.boxplot(x = 'Clusters', y= N, data  = h_rfm.KM())
    st.pyplot(ee)

  def scatter_Cluster(self):
    c = plt.figure(figsize = (20,15))
    a= st.sidebar.radio(label='SELECT FEATURE For Clustering', options=('Recency','Frequency','Monetary','Monetary_T','Frequency_T','Recency_T'), key = '2f6gd5d455')
    b = st.sidebar.radio(label='SELECT FEATURE For  Clustering', options=('Recency','Frequency','Monetary','Monetary_T','Frequency_T','Recency_T'), key = '2f6gd5d4555gd5')
    plt.scatter(x = h_rfm.KM()[a], y= h_rfm.KM()[b],c =h_rfm.KM()['Clusters'] , s = 200, cmap = 'viridis', edgecolor = 'black')
    plt.grid(color  = 'black', linewidth = 0.5)
#     plt.title('Clustering', Fontsize = 15)
#     plt.xlabel(a, Fontsize = 15)
#     plt.ylabel(b, Fontsize = 15)
    st.pyplot(c)
  def Access_hybrid(self):
    CI= st.number_input('Enter a Customer ID',0, 9000000, 0, 1, key = '100')
    for i in range(len( Hybrid_Analysis().KM())):
          if h_rfm.KM().CustomerID[i] == CI:
                st.write('Clusters   :', h_rfm.KM().Clusters[i])
                st.write('Recency :',h_rfm.KM().Recency[i])
                st.write('Frequency:',h_rfm.KM().Frequency[i])
                st.write('Monetary:',h_rfm.KM().Monetary[i])
          

#CLASSIFIER:#####################################################################
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

class ML_models():

  def inp_classifier(self):
    dff1['Bill'] = dff1['Bill'].astype(np.int64)
    dff1['BillDate'] = pd.to_datetime(dff1['BillDate'])
    dff1['CustomerID'] = dff1['CustomerID'].astype(int)
    Last_Date = dff1['BillDate'].max()
    RFM_table = dff1.groupby(by = ['CustomerID'], as_index = False).agg({'BillDate':lambda x: (Last_Date- x.max()).days,'Bill': lambda x: len(x), 'TotalAmount':lambda x: x.sum()})
    RFM_table.rename(columns = {'BillDate':'Recency', 'Bill':'Frequency', 'TotalAmount':'Monetary'}, inplace = True)

    RFM_table["Recency_Score"] = pd.qcut(RFM_table['Recency'], 5, labels=[5, 4, 3, 2, 1])
    RFM_table["Frequency_Score"] = pd.qcut(RFM_table["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    RFM_table["Monetary_Score"] = pd.qcut(RFM_table["Monetary"], 5, labels=[1, 2, 3, 4, 5])
    RFM_table["RFM_Score"] = (RFM_table["Recency_Score"].astype(str) + RFM_table["Frequency_Score"].astype(str))

    # seg_map = {r'[1-2][1-2]': 'Hibernating',r'[1-2][3-4]': 'At_Risk',r'[1-2]5': 'Cant_loose',r'3[1-2]': 'About_to_sleep',r'33': 'Need_attention',r'[3-4][4-5]': 'Loyal_customers',r'41': 'Promising',r'51': 'New_customers',r'[4-5][2-3]': 'Potential_loyalists',r'5[4-5]': 'Champions'}
    # RFM_table['Segment'] = RFM_table['RFM_Score'].replace(seg_map, regex=True) 

    RFM_h1 = rfm.RFMvalues().filter(['CustomerID', 'Recency', 'Frequency', 'Monetary'])
    RFM_h1['r_quartile'] = pd.qcut(RFM_h1['Recency'],5,labels  =  [5,4,3,2,1])
    RFM_h1['f_quartile'] = pd.qcut(RFM_h1['Frequency'].rank(method="first"), 5,labels =  [1,2,3,4,5])
    RFM_h1['rev_quartile'] = pd.qcut(RFM_h1['Monetary'],5,labels = [1,2,3,4,5])
    RFM_h1['Monetary_T'] = np.log10(RFM_h1['Monetary'])
    RFM_h1['Frequency_T'] = np.cbrt(RFM_h1['Frequency'])
    RFM_h1['Frequency_T'] = np.cbrt(RFM_h1['Frequency_T'])
    RFM_h1['Frequency_T'] = np.sqrt(RFM_h1['Frequency_T'])
    RFM_h1['Recency_T'] = np.cbrt(RFM_h1['Recency'])   
    return RFM_h1

  def accuracy_ML(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)

    svc = svm.SVC()
    parameters = [{'C':np.logspace(-2,2,10)}]
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, Y_train)
    GridSearchCV(estimator=SVC(),param_grid={'rf__max_depth': [4, 5, 10],'rf__max_features': [2, 3],'rf__min_samples_leaf': [3, 4, 5],'rf__n_estimators': [100, 200, 300]})
    Y_pred = clf.predict(X_test)
    accuracy_SVM = 100*metrics.accuracy_score(Y_test, Y_pred)

    
    lr = LogisticRegression()   
    lr.fit(X_train, Y_train)
    GridSearchCV(estimator=lr,param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    Y_pred1 = lr.predict(X_test)
    accuracy_Logres = 100*metrics.accuracy_score(Y_test, Y_pred1)



    knn = neighbors.KNeighborsClassifier()
    clf = KNeighborsClassifier(n_neighbors=5)
    # clf = GridSearchCV(knn, param_grid = np.arange(1,50,1), cv=10, scoring='accuracy', return_train_score=False,verbose=1)
    clf.fit(X_train, Y_train)
    GridSearchCV(estimator= knn ,param_grid={'rf__max_depth': [4, 5, 10],'rf__max_features': [2, 3],'rf__min_samples_leaf': [3, 4, 5],'rf__n_estimators': [100, 200, 300]})
    Y_pred2 = clf.predict(X_test)
    accuracy_KNN = 100*metrics.accuracy_score(Y_test, Y_pred2)


    parameters = [{'C':np.logspace(-2,2,10)}]
    p = tree.DecisionTreeClassifier()
    p.fit(X_train, Y_train)
    GridSearchCV(estimator=p ,param_grid={'criterion' : ['entropy', 'gini'], 'max_features' :['sqrt', 'log2']})
    Y_pred3 = p.predict(X_test)
    accuracy_Dec = 100*metrics.accuracy_score(Y_test, Y_pred3)

    rf = ensemble.RandomForestClassifier()  
    rf.fit(X_train, Y_train)
    GridSearchCV(estimator=rf,param_grid={'criterion' : ['entropy', 'gini'], 'n_estimators' : [20, 40, 60, 80, 100],'max_features' :['sqrt', 'log2']})
    Y_pred4 = rf.predict(X_test)
    accuracy_rf = 100*metrics.accuracy_score(Y_test, Y_pred4)

    AC = AdaBoostClassifier()
    AC.fit(X_train, Y_train)
    GridSearchCV(estimator=AC,param_grid={'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    Y_pred5 = AC.predict(X_test)
    accuracy_AC = 100*metrics.accuracy_score(Y_test, Y_pred5)

    gb = ensemble.GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    GridSearchCV(estimator=gb,param_grid={'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    Y_pred6 = gb.predict(X_test)
    accuracy_gb = 100*metrics.accuracy_score(Y_test, Y_pred6)

    rf_best  = ensemble.RandomForestClassifier()
    gb_best  = ensemble.GradientBoostingClassifier()
    svc_best = svm.LinearSVC()
    tr_best  = tree.DecisionTreeClassifier()
    knn_best = neighbors.KNeighborsClassifier()
    lr_best  = linear_model.LogisticRegression()

    votingC = ensemble.VotingClassifier(estimators=[('rf', rf_best),('gb', gb_best),
                                                ('knn', knn_best)], voting='soft')
    votingC = votingC.fit(X_train, Y_train)
    predictions = votingC.predict(X_test)
    accuracy_VC = 100*metrics.accuracy_score(Y_test, predictions)

    st.write('Accuracy of Support Vector Machine Model: ', accuracy_SVM )
    st.write('Accuracy of Logistc regression Model: ', accuracy_Logres )
    st.write('Accuracy of KNN Model: ', accuracy_KNN)
    st.write('Accuracy of Decision Tree Model: ',accuracy_Dec )
    st.write('Accuracy of Random Forest Model: ', accuracy_rf )
    st.write('Accuracy of AdaBoost Classifier Model: ',accuracy_AC )
    st.write('Accuracy of Gradient Booster Classfifier Model :', accuracy_gb)
    st.write('Accuracy of Soft Vote Classifier: ',accuracy_VC)


  def support(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)

    svc = svm.SVC()
    parameters = [{'C':np.logspace(-2,2,10)}]
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, Y_train)
    GridSearchCV(estimator=SVC(),param_grid={'rf__max_depth': [4, 5, 10],'rf__max_features': [2, 3],'rf__min_samples_leaf': [3, 4, 5],'rf__n_estimators': [100, 200, 300]})
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns]
    Y_pred = clf.predict(X_test)

    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']

    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred  
    # X_test['Clusters_Pred'] = Y_pred.labels_
    
    return Original_inp

  def Logistic_reg(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)

    svc = svm.SVC()
    parameters = [{'C':np.logspace(-2,2,10)}]
    clf = GridSearchCV(svc, parameters)
    clf.fit(X_train, Y_train)
    GridSearchCV(estimator=LogisticRegression(),param_grid={'C': [1, 10], 'kernel': ('linear', 'rbf')})
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns]
    Y_pred = clf.predict(X_test)

    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']

    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred

    return Original_inp
  def KNN(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)
    knn = neighbors.KNeighborsClassifier()
    clf = KNeighborsClassifier(n_neighbors=5)
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns] 
    # clf = GridSearchCV(knn, param_grid = np.arange(1,50,1), cv=10, scoring='accuracy', return_train_score=False,verbose=1)
    clf.fit(X_train, Y_train)
    GridSearchCV(estimator= knn ,param_grid={'rf__max_depth': [4, 5, 10],'rf__max_features': [2, 3],'rf__min_samples_leaf': [3, 4, 5],'rf__n_estimators': [100, 200, 300]})
    Y_pred = clf.predict(X_test)
    data = X_test['Predictions'] = Y_pred  
    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']
    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred
    return Original_inp

  def Dec_Tree(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)
    tr = tree.DecisionTreeClassifier()

    # tr.grid_search(parameters = [{'criterion' : ['entropy', 'gini'], 'max_features' :['sqrt', 'log2']}], Kfold = 5)
    k = GridSearchCV(estimator=tree.DecisionTreeClassifier() ,param_grid=[{'criterion' : ['entropy', 'gini'], 'max_features' :['sqrt', 'log2']}])
    k.fit(X_train, Y_train)
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns] 

    Y_pred = tr.predict(X_test)
    X_test['Predictions'] = Y_pred  
    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']
    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred
    return Original_inp

  def Random_F(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)
    rf = ensemble.RandomForestClassifier()  
    rf.fit(X_train, Y_train)
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns] 

    GridSearchCV(estimator=rf,param_grid={'criterion' : ['entropy', 'gini'], 'n_estimators' : [20, 40, 60, 80, 100],'max_features' :['sqrt', 'log2']})
    Y_pred = rf.predict(X_test)
    X_test['Predictions'] = Y_pred  
    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']
    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred
    return Original_inp

  def Adaboost(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)

    AC = AdaBoostClassifier()
    AC.fit(X_train, Y_train)
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns]
    GridSearchCV(estimator=AC,param_grid={'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    Y_pred5 = AC.predict(X_test)
    X_test['Predictions'] = Y_pred5  
    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']
    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred5
    return Original_inp
  def Graddient(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)

    gb = ensemble.GradientBoostingClassifier()
    gb.fit(X_train, Y_train)
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns]
    GridSearchCV(estimator=gb,param_grid={'n_estimators' : [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]})
    Y_pred6 = gb.predict(X_test)
    X_test['Predictions'] = Y_pred6 
    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']
    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = Y_pred6
    return Original_inp

  def Softvote(self):
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X = h_rfm.KM()[columns]
    Y = h_rfm.KM()['Clusters']
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, train_size = 0.8)

    rf_best  = ensemble.RandomForestClassifier()
    gb_best  = ensemble.GradientBoostingClassifier()
    svc_best = svm.LinearSVC()
    tr_best  = tree.DecisionTreeClassifier()
    knn_best = neighbors.KNeighborsClassifier()
    lr_best  = linear_model.LogisticRegression()

    votingC = ensemble.VotingClassifier(estimators=[('rf', rf_best),('gb', gb_best),
                                                ('knn', knn_best)], voting='soft')
    votingC = votingC.fit(X_train, Y_train)
    columns = ['Monetary_T','Frequency_T','Recency_T']
    X_test = ll.inp_classifier()[columns]
    predictions = votingC.predict(X_test)
  
    X_test['Predictions'] = predictions
    col = ['CustomerID', 'Monetary_T','Frequency_T','Recency_T']
    Original_inp = ll.inp_classifier()[col]
    Original_inp['Predictions'] = predictions
    return Original_inp

########################################################################################################

from lifetimes.utils import *
from lifetimes import BetaGeoFitter
from lifetimes.plotting import plot_probability_alive_matrix, plot_frequency_recency_matrix
from lifetimes.generate_data import beta_geometric_nbd_model
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_calibration_purchases_vs_holdout_purchases, plot_period_transactions,plot_history_alive
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler



class cltv():
  def Access_cltv(self):
    Num= st.number_input('Enter a Customer ID',0, 300000, 0, 1, key = '3063')
    for i in range(len(cc.C_L_T_V())):
      if cc.C_L_T_V().CustomerID[i]== Num:
        st.write('Segment :', cc.C_L_T_V().Segment[i])
        st.write('Expected Purchase of 1 week :',cc.C_L_T_V().expected_purchase_1_week[i])
        st.write('Expected Purchase of 1 Month :',cc.C_L_T_V().expected_purchase_1_month[i])
        st.write('Expected Average Profit :',cc.C_L_T_V().expected_average_profit[i])
        st.write('Customer CLV Value:',cc.C_L_T_V().clv[i])
        break


  def C_L_T_V(self):
    today_date = dt.datetime(2019, 12, 10)
    # cltv_df = dff.groupby('CustomerID').agg({'BillDate': [lambda date: (date.max() - date.min()).days,
    #                                                      lambda date: (today_date - date.min()).days],
    #                                      'Bill': lambda num: num.nunique(),
    #                                      'Amount': lambda TotalPrice: TotalPrice.sum()})
    necessary_cols = ['CustomerID', 'BillDate', 'TotalAmount']
    cltv_df = dff[necessary_cols]
    last_order_date = cltv_df['BillDate'].max()
    cltv_data = summary_data_from_transaction_data(cltv_df, 'CustomerID', 'BillDate', monetary_value_col='TotalAmount', observation_period_end='2019-10-07')
    


    cltv_data["monetary_value"] = cltv_data["monetary_value"] / cltv_data["frequency"]
    cltv_data = cltv_data[cltv_data["monetary_value"] > 0]
    cltv_data["recency"] = cltv_data["recency"] / 7
    cltv_data["T"] = cltv_data["T"] / 7
    cltv_data = cltv_data[(cltv_data['frequency'] > 1)]



    bgf = BetaGeoFitter(penalizer_coef=100)
    # plot_period_transactions(bgf) #Plot of frequency of repeat transactions.........(TO SHOW) (will be like ##1 Image refer)
    

    bgf.fit(cltv_data['frequency'], cltv_data['recency'], cltv_data['T'])
    #------------------------------------------------------------
#     zz  =plt.figure(figsize = (15,15))
#     plot_period_transactions(bgf) 
    
#     st.pyplot(zz)

    #-------------------------------------
    cltv_data["expected_purchase_1_week"] = bgf.predict(1,cltv_data['frequency'],cltv_data['recency'],cltv_data['T'])
    cltv_data["expected_purchase_1_month"] = bgf.predict(4,cltv_data['frequency'],cltv_data['recency'],cltv_data['T'])


    ggf = GammaGammaFitter(penalizer_coef=100)
    ggf.fit(cltv_data['frequency'], cltv_data['monetary_value'])
    
    ggf.summary
    cltv_data["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_data['frequency'],cltv_data['monetary_value'])
    cltv = ggf.customer_lifetime_value(bgf,cltv_data['frequency'],cltv_data['recency'],cltv_data['T'],cltv_data['monetary_value'],time=6, discount_rate=0.01)
    cltv = cltv.reset_index()
    cltv_final = cltv_data.merge(cltv, on="CustomerID", how="left")
    cltv_1 = ggf.customer_lifetime_value(bgf,cltv_data['frequency'],cltv_data['recency'], cltv_data['T'],cltv_data['monetary_value'], time=1,  freq="W",   discount_rate=0.01)
    cltv_1= cltv_1.reset_index()
    cltv_1 = cltv_data.merge(cltv_1, on="CustomerID", how="left")
    cltv_12 = ggf.customer_lifetime_value(bgf,
                                   cltv_data['frequency'],
                                   cltv_data['recency'],
                                   cltv_data['T'],
                                   cltv_data['monetary_value'],
                                   time=12,  # 12 months
                                   freq="W",  # Frequency of T
                                   discount_rate=0.01)
    cltv_12 = cltv_12.reset_index()
    cltv_12 = cltv_data.merge(cltv_12, on="CustomerID", how="left")
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(cltv_final[["clv"]])
    cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])
    cltv_final["Segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])

    # return cltv_final.astype('object')
    return cltv_final.astype('object')
     
###################################################################################################

class Product_Recommendation():
  
  def Association_rules(self):
    dff['Product'] = dff['Product'].str.strip()
    # mybasket_UK = dff[dff['Country'] == 'United Kingdom'].groupby(['Bill', 'Product'])['Quota'].sum().unstack().reset_index().fillna(0).set_index('Bill')
    mybasket_UK = dff.groupby(['Bill', 'Product'])['Quota'].sum().unstack().reset_index().fillna(0).set_index('Bill')
    def rr(x):
      if x<=0:
        return 0
      if x>= 1:
        return 1
    mybasket_UK_sets = mybasket_UK.applymap(rr)
    
    my_frequent_products = apriori(mybasket_UK_sets, min_support = 0.001, use_colnames = True )
    my_rules_UK = association_rules(my_frequent_products,metric = 'lift', min_threshold = 0.2)
    my_rules_UK = pd.DataFrame(my_rules_UK)
    # st.write(my_rules_UK)
    st.table(my_rules_UK.applymap(lambda x: tuple(x) if isinstance(x, frozenset) else x ))
    st.write('''Above table represents to the recommendation scenario and association between your products as per the dataset provided.An empty table represents that the data necessary to deduce recommendation is insufficient or no recommendation can take place as per the dataset provided.
	For a filled dataset, the value of support indicates the popularity of a single product based on customer purchase behaviour. Accordingly, you must try to set or redefine the prices of the respective products to grab customers as well as profit. Confidence in the above table indicates a measurement of how often customers purchase two products in an item set. A higher confidence value shows that a customer is more likely to buy the second item when they buy the first item.In the above table, the products corresponding to lift value greater than 1 are closely associated and customers are likely to purchase those products together.
''')
########################################################################################################

class Sales_Forecasting():
  def LSTM_Model(self):
    #Shaping the given data for sales forecasting
    sales_rd= dff.copy(deep=True)
    sales_rd=sales_rd.dropna()
    cols = ['Bill',	'MerchandiseID',	'Product',	'Quota',	'CustomerID',	'Country',	'Amount']
    sales_rd.drop(cols, axis=1, inplace=True)
    sales_rd = sales_rd.sort_values('BillDate')
    sales_rd_02 = sales_rd.groupby('BillDate')['TotalAmount'].sum().reset_index()

    #Seperating train and test from given data
    sales_data = sales_rd_02['TotalAmount'].values
    sales_data = sales_data.reshape((-1,1))

    split_percent = 0.8
    split = int(split_percent*len(sales_data))

    sales_train = sales_data[:split]
    sales_test = sales_data[split:]

    date_train = sales_rd_02['BillDate'][:split]
    date_test = sales_rd_02['BillDate'][split:]

    print(len(sales_train))
    print(len(sales_test))

    look_back = 5    #can be varied for accuracy

    train_generator = TimeseriesGenerator(sales_train, sales_train, length=look_back, batch_size=5)     
    test_generator = TimeseriesGenerator(sales_test, sales_test, length=look_back, batch_size=1)

    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    model = Sequential()
    model.add(
        LSTM(10,
            activation='relu',
            input_shape=(look_back,1))
    )
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    num_epochs = 300   #can be varied for accuracy
    model.fit_generator(train_generator, epochs=num_epochs, verbose=1)

    prediction = model.predict_generator(test_generator)

    sales_train = sales_train.reshape((-1))
    sales_test = sales_test.reshape((-1))
    prediction = prediction.reshape((-1))

    #actual= sales_test[:116]   #check again

    sales_data = sales_data.reshape((-1))
    def predict(num_prediction, model):
        prediction_list = sales_data[-look_back:]
        
        for _ in range(num_prediction):
            x = prediction_list[-look_back:]
            x = x.reshape((1, look_back, 1))
            out = model.predict(x)[0][0]
            prediction_list = np.append(prediction_list, out)
        prediction_list = prediction_list[look_back-1:]
            
        return prediction_list

    def predict_dates(num_prediction):
        last_date = sales_rd_02['BillDate'].values[-1]
        prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
        return prediction_dates

    num_prediction = 30     #forecasting for 30 more dates from the last Bill Date
    forecast = predict(num_prediction, model)
    forecast_dates = predict_dates(num_prediction)

    forecasted_sales_data = pd.DataFrame(
    {'BillDate': forecast_dates,
     'PredictedAmount': forecast,
     })
    
    return forecasted_sales_data.reset_index()

  def Access_Forecast(self):
    Num= st.number_input('Enter a particular number in the forecasted month',0, 30, 0, 1, key = '30')
    for i in range(len(sf.LSTM_Model())):
      if sf.LSTM_Model().index[i]== Num:
        st.write('DATE & TIME :', sf.LSTM_Model().BillDate[i])
        st.write('FORECASTED SALES :',sf.LSTM_Model().PredictedAmount[i])
        break

class Churn_Analysis():

  def coh(self):
    from datetime import datetime
    dff['BillDate'] = pd.to_datetime(dff['BillDate'])
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    totalamount_df = dff.groupby(['BillYearMonth'])['TotalAmount'].sum().reset_index()
    #Required code to run plotly in the cell
    # configure_plotly_browser_state()
#     init_notebook_mode(connected=False)
    #X and Y axis inputs for Plotly graph. We use Scatter for line graphs
    plot_data = [
    go.Scatter(x=totalamount_df['BillYearMonth'],y=totalamount_df['TotalAmount'],)]
    plot_layout = go.Layout(xaxis={"type": "category"},title='Monthly Revenue', autosize= True)
    fig = go.Figure(data=plot_data, layout=plot_layout)
    # pyoff.iplot(fig)
    st.plotly_chart(fig)
  def coh2(self):
    from datetime import datetime
    dff['BillDate'] = pd.to_datetime(dff['BillDate'])
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    totalamount_df = dff.groupby(['BillYearMonth'])['TotalAmount'].sum().reset_index()
    #using pct_change() function to see monthly percentage change
    totalamount_df['MonthlyGrowth'] = totalamount_df['TotalAmount'].pct_change()
    #showing first 5 rows
    totalamount_df.head()
    #visualization - line graph
    plot_data = [go.Scatter(x=totalamount_df.query("BillYearMonth < 201910")['BillYearMonth'],y=totalamount_df.query("BillYearMonth < 201910")['MonthlyGrowth'],)]
    plot_layout = go.Layout(xaxis={"type": "category"},title='Montly Growth Rate')
    fig1 = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig1)
  def coh3(self):
    from datetime import datetime
    dff['BillDate'] = pd.to_datetime(dff['BillDate'])
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_monthly_active = tx_uk.groupby('BillYearMonth')['CustomerID'].nunique().reset_index()
    plot_data = [go.Bar(x=tx_monthly_active['BillYearMonth'],y=tx_monthly_active['CustomerID'],)]
    plot_layout = go.Layout(xaxis={"type": "category"},title='Monthly Active Customers')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
  def coh4(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_monthly_sales = tx_uk.groupby('BillYearMonth')['Quota'].sum().reset_index()
    #plot
    plot_data = [go.Bar(x=tx_monthly_sales['BillYearMonth'],y=tx_monthly_sales['Quota'],)]
    plot_layout = go.Layout(xaxis={"type": "category"},title='Monthly Total # of Order')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
  def coh5(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_monthly_order_avg = tx_uk.groupby('BillYearMonth')['TotalAmount'].mean().reset_index()
    tx_monthly_order_avg.rename(columns={'TotalAmount': 'AverageAmount'}, inplace=True)
    #plot the bar chart
    plot_data = [
    go.Bar(x=tx_monthly_order_avg['BillYearMonth'],y=tx_monthly_order_avg['AverageAmount'],)]
    plot_layout = go.Layout(xaxis={"type": "category"},title='Monthly Order Average')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
  def coh6(self):

    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_min_purchase = tx_uk.groupby('CustomerID').BillDate.min().reset_index()
    tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']
    tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)
    #merge first purchase date column to our main dataframe (tx_uk)
    tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')
    tx_uk['UserType'] = 'New'
    tx_uk.loc[tx_uk['BillYearMonth']>tx_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'
    tx_user_type_revenue = tx_uk.groupby(['BillYearMonth','UserType'])['TotalAmount'].sum().reset_index()
    plot_data = [go.Scatter(x=tx_user_type_revenue.query("UserType == 'Existing'")['BillYearMonth'],y=tx_user_type_revenue.query("UserType == 'Existing'")['TotalAmount'],name = 'Existing'),go.Scatter(x=tx_user_type_revenue.query("UserType == 'New'")['BillYearMonth'],y=tx_user_type_revenue.query("UserType == 'New'")['TotalAmount'],name = 'New')]
    plot_layout = go.Layout(xaxis={"type": "category"},title='New vs Existing')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)

  def coh7(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_uk['UserType'] = 'New'
    tx_min_purchase = tx_uk.groupby('CustomerID').BillDate.min().reset_index()
    tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']
    tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)
    #merge first purchase date column to our main dataframe (tx_uk)
    tx_uk = pd.merge(tx_uk, tx_min_purchase, on='CustomerID')
    tx_uk.loc[tx_uk['BillYearMonth']>tx_uk['MinPurchaseYearMonth'],'UserType'] = 'Existing'

    tx_user_ratio = tx_uk.query("UserType == 'New'").groupby(['BillYearMonth'])['CustomerID'].nunique()/tx_uk.query("UserType == 'Existing'").groupby(['BillYearMonth'])['CustomerID'].nunique() 
    tx_user_ratio = tx_user_ratio.reset_index()
    tx_user_ratio = tx_user_ratio.dropna()

    plot_data = [go.Bar(x=tx_user_ratio.query("BillYearMonth>201712 and BillYearMonth<201910")['BillYearMonth'],y=tx_user_ratio.query("BillYearMonth>201712 and BillYearMonth<201910")['CustomerID'],)]
    plot_layout = go.Layout(xaxis={"type": "category"},title='New Customer Ratio')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
  def coh9(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_monthly_active = tx_uk.groupby('BillYearMonth')['CustomerID'].nunique().reset_index()
    return tx_monthly_active

  def coh8(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_user_purchase = tx_uk.groupby(['CustomerID','BillYearMonth'])['TotalAmount'].sum().reset_index()
    #create retention matrix with crosstab
    tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['BillYearMonth']).reset_index()
    months = tx_retention.columns[2:]
    retention_array = []
    for i in range(len(months)-1):
      retention_data = {}
      selected_month = months[i+1]
      prev_month = months[i]
      retention_data['BillYearMonth'] = int(selected_month)
      retention_data['TotalUserCount'] = tx_retention[selected_month].sum()
      retention_data['RetainedUserCount'] = tx_retention[(tx_retention[selected_month]>0) & (tx_retention[prev_month]>0)][selected_month].sum()
      retention_array.append(retention_data)
    tx_retention = pd.DataFrame(retention_array)
    tx_retention['RetentionRate'] = tx_retention['RetainedUserCount']/tx_retention['TotalUserCount']
    return tx_retention
  def coh_ret(self):
    plot_data = [go.Scatter(x=chu.coh8().query("BillYearMonth<201910")['BillYearMonth'],y=chu.coh8().query("BillYearMonth<201910")['RetentionRate'],name="organic")]
    plot_layout = go.Layout(xaxis={"type": "category"},title='Monthly Retention Rate')
    fig = go.Figure(data=plot_data, layout=plot_layout)
    st.plotly_chart(fig)
  def coh10(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_user_purchase = tx_uk.groupby(['CustomerID','BillYearMonth'])['TotalAmount'].sum().reset_index()
    tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['BillYearMonth']).reset_index()
    return tx_retention
  def coh11(self):
    dff['BillYearMonth'] = dff['BillDate'].map(lambda date: 100*date.year + date.month)
    tx_uk = dff.query("Country=='United Kingdom'").reset_index(drop=True)
    tx_user_purchase = tx_uk.groupby(['CustomerID','BillYearMonth'])['TotalAmount'].sum().reset_index()
    #create our retention table again with crosstab() and add firs purchase year month view
    tx_uk['UserType'] = 'New'
    tx_min_purchase = tx_uk.groupby('CustomerID').BillDate.min().reset_index()
    tx_min_purchase.columns = ['CustomerID','MinPurchaseDate']
    tx_min_purchase['MinPurchaseYearMonth'] = tx_min_purchase['MinPurchaseDate'].map(lambda date: 100*date.year + date.month)
    tx_retention = pd.crosstab(tx_user_purchase['CustomerID'], tx_user_purchase['BillYearMonth']).reset_index()
    months = tx_retention.columns[2:]
    tx_retention = pd.merge(tx_retention,tx_min_purchase[['CustomerID','MinPurchaseYearMonth']],on='CustomerID')
    new_column_names = [ 'm_' + str(column) for column in tx_retention.columns[:-1]]
    new_column_names.append('MinPurchaseYearMonth')
    tx_retention.columns = new_column_names
    # create the array of Retained users for each cohort monthly
    retention_array = []
    for i in range(len(months)):
        retention_data = {}
        selected_month = months[i]
        prev_months = months[:i]
        next_months = months[i+1:]
        for prev_month in prev_months:
	        retention_data[prev_month] = np.nan
		
        total_user_count = tx_retention[tx_retention.MinPurchaseYearMonth ==  selected_month].MinPurchaseYearMonth.count()
        retention_data['TotalUserCount'] = total_user_count
        retention_data[selected_month] = 1 
        query = "MinPurchaseYearMonth == {}".format(selected_month)
        for next_month in next_months:
	        new_query = query + " and {} > 0".format(str('m_' + str(next_month)))
	        retention_data[next_month] = np.round(tx_retention.query(new_query)['m_' + str(next_month)].sum()/total_user_count,2)
        retention_array.append(retention_data)
	
    tx_retention = pd.DataFrame(retention_array)
    tx_retention.index = months
    return tx_retention



###################################################################################
class About():
	
	def Dev(self):
		st.subheader('About Organization')
		colf, colg = st.columns([2.5,1])
		with colf:
			st.write('''Hey user, meet the team behind the creation of this Customer Relationship Management Tool. Every individual of TMA keeps the vision to apply his learnings and engineering skills towards innovation, updation and revolution.
This interface is the first step of TMA in the field of ‘Predictive Analytics’, which is a promising domain and is expected to drive the world in near future.
The Team Mettle Amigos, consisting of tech enthusiasts and innovative minds, with the happy faces in the frame below, have many future prospects and this interface is just a start of it.''')
		with colg:
			imgs = Image.open("./images/TMA-01.png")
			st.image(imgs,caption = 'TEAM METTLE AMIGOS',use_column_width= None )
		st.subheader('About Team members')
		colh, coli, colj = st.columns(3)
		
		with colh:
			imgsa = Image.open("./images/tma_images/3.png")
			st.image(imgsa,use_column_width= None )
		with coli:
			imgsb = Image.open("./images/tma_images/4.png")
			st.image(imgsb,use_column_width= None )
		with colj:
			imgsc = Image.open("./images/tma_images/5.png")
			st.image(imgsc,use_column_width= None )
		colk, coll,colm,coln,colo = st.columns([0.111,0.3,0.111,0.3,0.111])
		with coll:
			imgsd = Image.open("./images/tma_images/6.png")
			st.image(imgsd,use_column_width= None )
		with coln:
			imgse = Image.open("./images/tma_images/7.png")
			st.image(imgse,use_column_width= None )
		st.subheader('Contact Team Mettle Amigos')
		st.write('''For any queries, get in touch with us at teammettleamigos5@gmail.com .
To contact any individual from the team, you can directly message the particular individual over linkedin by searching his name.
''')
		
		
		
		
		
		
		
	def User(self):
		
		with st.expander('HOME'):
			st.write(''' Welcome to the CRM Interface! To help guide you through all the nitty gritty of the Interface this
Home Page makes it easy to make you and our Interface more compatible.
In this section, you will be exploring on how to be adaptive to the CRM Interface by understanding
a brief overview of all the features, quick-go through video, options for uploading data sets and
the other options directing you to access all the feature-activities.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Get a brief overview of the CRM Interface')
			st.write('2.Created structured format of Main and Prediction datasets')
			st.write('3.Upload Main and Prediction datasets in the CRM Interface')
			st.write('4.Display and get access to the Input of Main data')
			st.write('5.Display and get access to the Input of Prediction data')
			
			cola1,cola2,cola3 = st.columns([0.1,2,0.1])
			with cola2:
				HtmlFile_1 = open("images/UserGuide./Home-Page_Compressed.html", 'r', encoding='utf-8')
				source_code1 = HtmlFile_1.read()
				print(source_code1)
				components.html(source_code1, height = 700) 							
			
		with st.expander('EDA ANALYSIS'):
			st.write('''Visualization of your input dataset will help you with your analysis. To understand the statistical
data, visualization and correlations of the attributes, and hidden patterns we provide you the
feature-activity tool of EDA Analysis in CRM Interface which will exclusively help you analyse and
visualize your data set.In this section, you will be exploring on how to use three options provided under EDA Analysis
which are analysis through SweetViz, Pandas Profiling and understanding the demographics of
your customers. These three options will help you with analysis and visualization of your datasets
so as to get a comprehensive understanding.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Access the analysis and visualization reports through Sweetviz')
			st.write('2.Access the analysis and visualization reports through Pandas Profiling')
			st.write('3.Access the visualization of demographics of your customers')
			
			colb1, colb2, colb3 = st.columns([0.1,2,0.1])
			with colb2:
				HtmlFile_2 = open("images/UserGuide./EDA-Analysis_Compressed.html", 'r', encoding='utf-8')
				source_code2 = HtmlFile_2.read()
				print(source_code2)
				components.html(source_code2, height = 700)				
				
		with st.expander('CUSTOMER SEGMENTATION'):
			st.write(''' Segmenting customers is the process of dividing up mass consumers into groups with similar
needs and wants. It also helps in attaining customer satisfaction and overall profit at higher rates.
Also helps organizations to focus on efficient resource allocation. But there exists a very common
axiom "One Size Fits All", i.e. treating every customer in the same manner. From a customer point
of view it is courteous, but from a business point of view it is not profitable at all. That's why
segmentation exists to mitigate this problem. There exists 6 types of customer segmentation so
far which are Geographic, Demographic, Behavioral, Firmographic, Psychographic and smart
customer segmentation. Here, in the CRM Interface segmentation of customers is done on the
basis of RFM and Hybrid (K-means with RFM) modeling techniques. These feature-activities will
help you with segmenting the customers and then further with the help of it one can decide the
strategies.
In this section, you will be exploring on how to perform customer segmentation using RFM and
Hybrid (K-means with RFM) modeling techniques.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Upload the dataset in the interface')
			st.write('2.Use the different features related to both modelling techniques')
			st.write('3.Acquire customer segmentation results')
			st.write('4.Download the relevant processed table as a CSV file')
			st.write('5.Access the segmented regions of customers')
			colc1, colc2, colc3 = st.columns([0.1,2,0.1])
			with colc2:
				st.subheader('RFM SEGMENTATION')
				HtmlFile_3 = open("images/UserGuide./Customer Segmentation_01_RFM_Compressed.html", 'r', encoding='utf-8')
				source_code3 = HtmlFile_3.read()
				print(source_code3)
				components.html(source_code3, height = 700)
				
				st.subheader('HYBRID SEGMENTATION')
				HtmlFile_3a = open("images/UserGuide./Customer Segmentation_02_Hybrid_Compressed.html", 'r', encoding='utf-8')
				source_code3a = HtmlFile_3a.read()
				print(source_code3a)
				components.html(source_code3a, height = 700)
				
				
				
				
				
				
				
				
		with st.expander('CUSTOMER CLASSIFICATION'):
			st.write(''' A classifier in general is any algorithm that sorts data into labelled classes, or categories of
information. A simple practical example are spam filters that scan incoming “raw” emails and
classify them as either “spam” or “not-spam.” Classifiers are a concrete implementation of
pattern recognition in many forms of machine learning.
Here, classifier for the model will help you classify the customers into various categories. For this,
analysis and validation of different classifiers like Support Vector Classifier (SVC), Logistic
Regression, K Nearest neighbors Classifier, Decision Tree, Random Forest, AdaBoost Classifier and
Gradient Boosting Classifier is done to let you select the right classifier based on its predicting
ability, quality of fit and your requirements.
In this section, you will be exploring on how to perform customer classification.''')
			st.subheader('In this section, you will learn how to')
			st.write('Display accuracy of all classification models')
			st.write('Choose the right classifier amongst different classification models for prediction')			
			cold1, cold2, cold3 = st.columns([0.1,2,0.1])
			with cold2:
				HtmlFile_4 = open("images/UserGuide./Customer Classification_Compressed.html", 'r', encoding='utf-8')
				source_code4 = HtmlFile_4.read()
				print(source_code4)
				components.html(source_code4, height = 700)
				
				
				
				
		with st.expander('SALES FORECASTING'):
			st.write(''' Sales forecasting is the business methodology of predicting future sales of a company for a
certain time period (daily, weekly, monthly, quarterly, etc.). Though it’s hard to forecast the
accurate sales, there are certain AI-ML methodologies using which the sales can be predicted for
business analysis. Forecasts are helpful to most of the departments in any company like sales,
operations, productions departments, etc.
To forecast the future sales based on historic sales patterns in the dataset, the time series analysis
of data is done. This analysis comprises of extracting meaningful statistics and other
characteristics of the data. In this interface, forecasting of future sales using the historical sales
patterns in the dataset is done by using time series forecasting models.
In this section, you will be exploring on how to perform sales forecasting and access results in
different ways on this interface.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Upload the dataset in the interface')
			st.write('2.Use the different features and acquire the forecasts')
			st.write('3.Download the Sales forecasting table as a CSV file')
			st.write('Access the forecasting of a certain time period')
			cole1, cole2, cole3 = st.columns([0.1,2,0.1])
			with cole2:
				HtmlFile_5 = open("images/UserGuide./Sales Forecasting_Compressed.html", 'r', encoding='utf-8')
				source_code5 = HtmlFile_5.read()
				print(source_code5)
				components.html(source_code5, height = 700)
							
				
		with st.expander('PRODUCT RECOMMENDATION'):
			st.write(''' Product Recommendation, also known as market basket analysis, is a method for predicting
which product combinations will sell the best based on inventory and sales data. In our interface,
using association rules, the grouping of different products is done and the results are displayed.
Using this feature of our interface, businesses can then determine which products are frequently
purchased in conjunction with the solution.
In this section, you will be exploring on how to generate the data for product recommendation
and analyse it further for decision making.''')
			
			st.subheader('In this section, you will learn how to')
			st.write('1.Upload the dataset in the interface')
			st.write('2.Generate the association rules for recommending products')
			st.write('3.Analyse the product recommendation data')
			colf1, colf2, colf3 = st.columns([0.1,2,0.1])
			with colf2:
				HtmlFile_6 = open("images/UserGuide./Product Recommendation.html", 'r', encoding='utf-8')
				source_code6 = HtmlFile_6.read()
				print(source_code6)
				components.html(source_code6, height = 700)
				
				
				
		with st.expander('CHURN RATE ANALYSIS'):
			st.write(''' In this section, analysis of forensic aspects of a dataset like churn rate using different visual plots
and key metrics can be done. Analyzing forensic aspects is a step to offer customized solutions
and motivate actions. Forensic analysis helps businesses to identify deviations from the normal.
In our interface, by considering the input data, various visualisations can be displayed in this
section, using which the you can know the market trends and decide the customized solutions.
In this section, you will come across several such bar, line, scatter plots, etc. in the form of results
which can help you make sound business decisions.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Upload the dataset in the interface')
			st.write('2.Generate different visualisations related to forensic aspects')
			st.write('3.Analyse the visualisations for decision making')
			colg1,colg2, colg3  = st.columns([0.1,2,0.1])
			with colg2:
				HtmlFile_7 = open("images/UserGuide./Churn Rate Analysis_Compressed.html", 'r', encoding='utf-8')
				source_code7 = HtmlFile_7.read()
				print(source_code7)
				components.html(source_code7, height = 700)
				
				
				
				
		with st.expander('CLTV'):
			st.write(''' In this section, the focus is made on generating customers’ lifetime value for a business of the
provided dataset. CLTV simply is a predicted value of company's net profit contributed to its
overall future relationship with a customer. Knowing the customer base’s lifetime value can
enable businesses to strategize and better the customer experience accordingly, thus increasing
the profits of future. In our interface, by providing the input data, a tabulated information related
to various aspects of customers’ lifetime value can be generated.
In this section, you will come across the steps on how to predict customer’s lifetime value using
this interface.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Upload the dataset in the interface')
			st.write('2.Generate the information measuring customer’s value')
			st.write('3.Download the Sales forecasting table as a CSV file')
			st.write('4.Access the lifetime value and other aspects of a certain customer')
			
			colh1, colh2, colh3 = st.columns([0.1,2,0.1])
			with colh2:
				HtmlFile_8 = open("images/UserGuide./CLTV_Compressed.html", 'r', encoding='utf-8')
				source_code8 = HtmlFile_8.read()
				print(source_code8)
				components.html(source_code8, height = 700)
				
				
				
			
		with st.expander('ABOUT'):
			st.write(''' In this section, some easy steps to use and access the about page of this interface will be
discussed. The primary purpose of an about us page is to inform the reader, especially a beginner
user, about the company, its vision and more about website. In our interface, besides the general
information, we also provide the user steps for different features which can guide user for
smooth operations.
In this section, you will get a quick view on what exactly is the about page, its use for the
beginners and how your queries can be resolved.''')
			st.subheader('In this section, you will learn how to')
			st.write('1.Use the About Page in the interface')
			st.write('2.Know more about the inception of this interface')
			st.write('3.Get in touch with the team behind this interface')
			
			coli1, coli2, coli3 = st.columns([0.1,2,0.1])
			with coli2:
				HtmlFile_9 = open("images/UserGuide./About Page.html", 'r', encoding='utf-8')
				source_code9 = HtmlFile_9.read()
				print(source_code9)
				components.html(source_code9, height = 700)
	

				
				
				
				
	
	
	
	
	def Feed(self):
		HtmlFile_10 = open("Feedback./New_Customer_Registration_Form.html", 'r', encoding='utf-8')
		source_code10 = HtmlFile_10.read()
		print(source_code10)
		components.html(source_code10, height = 2000)
		
# 		import sqlite3
# 		conn = sqlite3.connect('student_feedback.db')
# 		c = conn.cursor()
# 		def create_table():
# 			c.execute('CREATE TABLE IF NOT EXISTS feedback(date_submitted DATE, Q1 TEXT, Q2 INTEGER, Q3 INTEGER, Q4 TEXT, Q5 TEXT, Q6 TEXT, Q7 TEXT, Q8 TEXT)')
# 		def add_feedback(date_submitted,Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8):
# 			c.execute('INSERT INTO feedback (date_submitted, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8) VALUES (?,?,?,?,?,?,?,?,?)',(date_submitted, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8))
# 			conn.commit()
# 		st.title("User Feedback")
# 		d = st.date_input("Today's date",None, None, None, None)
# 		question_1 = st.selectbox('Where did you got to know about these app?',('Youtube','Github', 'Linkedin', 'Twitter','Instagram','Recommended by friend'))
# 		st.write('You selected:', question_1)
		
# 		question_1 = st.text_input('Please Enter your Name?')
# 		st.write('Your Name is :', question_1)
		
# 		question_2 = st.slider('Please Enter your current age?', 18,100)
# 		st.write('You age:', question_2)
		
# 		question_3 = st.selectbox('Please Enter your Gender',('','Male', 'Women','Others'))
# 		st.write('Your Gender is:', question_3)
		
# 		question_4 = st.text_input('Please Enter Name of the Organization/Company/Brand')
# 		st.write('You work in:', question_4)
		
# 		question_5 = st.slider('Overall, how satisfied are you with the Application? (10 being very happy and 1 being very dissapointed)', 1,10,1)
# 		st.write('You selected:', question_5)
		
# 		question_6 = st.selectbox('Was the application fun and interactive?',('','Yes', 'No'))
# 		st.write('You selected:', question_6)
		
# 		question_7 = st.text_input('Please Enter your Email id/Contact number')
# 		st.write('You selected:', question_7)
		
		
# 		question_8 = st.text_input('Please Enter any Query/Suggestion/Complain/doubt if you have one.')
# 		st.write('You selected:', question_8)
		
		
		
# 		if st.button("Submit feedback"):
# 			create_table()
# 			add_feedback(d,question_1, question_2, question_3,question_4,question_5,  question_6, question_7, question_8)
# 			st.success("Feedback submitted")
# 			query = pd.read_sql_query('''select * from feedback''', conn)
# 			data = pd.DataFrame(query)
# 			st.write(data)
			
			
		



			
			
			
			
		
			
			
			
			
	
  
#------------------------------xox----------------------------------------------

def main():
  activities = [ 'HOME', 'Data Visualization and Analysis', 'Customer Segmentation','Customer Classification','Sales Forecasting','Product Recommendation','Forensic Analysis','Customer Linked Predictions', 'About']
  choice = st.sidebar.selectbox("Select Activities",activities, key = '6')

  if choice == 'HOME':
    st.subheader('**Welcome to the application!!! Visualize CRM tasks in one click.**')
    g.intro()
  if choice == 'Data Visualization and Analysis':
    st.sidebar.subheader('Exploratory Data Analysis')
    st.subheader('Welcome to the section of Exploratory Data Analysis!! Visualize your input dataset at just one click in a detailed report format')
    if st.sidebar.checkbox('SweetViz Analysis', key = '10'):
      st.write(dataframe.SweetV(dff))
      st.write('Here is the visualisation of your input dataset brought to you by SweetViz.Sweetviz is an open-source Python library that helps generate beautiful, highly detailed visualizations. The above generated report contains statistical data and corresponding visualizations of all the attributes of the dataset. In the above report, specially check for the missing and distinct values.')
      add_line1= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line1, unsafe_allow_html=True)
      dataframe.Exp1()
    if st.sidebar.checkbox('Pandas Profiling', key = '11'):
      st.subheader('Pandas Profiling Data Analysis Report')
      profile = ProfileReport(dff)
      st_profile_report(profile)
      profile.to_notebook_iframe()
      st.write('Here is your exploratory data analysis report brought to you by Pandas Profiling library in python. The above report contains overview information about the dataset given as input here. Don’t miss some important aspects included in this report like cardinality, correlation, variable types, etc.')
      add_line1= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line1, unsafe_allow_html=True)

    if st.sidebar.checkbox('Demographics of Customer'):
      st.subheader('Demographics of Customers')
      dataframe.Map()
      st.write('Here’s the demographic visualization of your customers globally. This map visualisation is brought to you by ‘Mapbox’, a powerful tool for building interactive customizable maps and integrating location and navigation data into your apps and websites. The regions highlighted in red in the map above denote the concentration of customers in that particular regions. Let’s hope someday that your business captures the whole globe and you may see this map whole in red!!')
      add_line1= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line1, unsafe_allow_html=True)
  if choice == 'Product Recommendation':
    rules.Association_rules()
    add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
    st.markdown(add_line, unsafe_allow_html=True)
  if choice == 'Customer Segmentation':
    st.sidebar.subheader('Choose Method for Customer Segmentation')
    st.write('Try our two best methods of segmenting customers, one via the RFM metrics and a unique hybrid method. Hope you get fascinating results about your customer base here!')
    if st.sidebar.checkbox('RFM_SEGMENTATION', key = '12'):
      st.subheader('RFM Segmentation Dataframe')
      st.download_button(label="Download data as CSV",data=rfm.RFMvalues().to_csv().encode('utf-8'),file_name='RFM_values.csv',mime='text/csv',)
      st.write(rfm.RFMvalues().astype('object'))
      st.write('Here’s the tabulated information of customers in form of their recency, frequency and monetary value i.e. RFM. Recency denotes the number of days since last activity of customer, while frequency denotes the number of purchases done till date and the monetary denotes the spending capability of the customers. Accordingly, your customers have been rated in range of 0-5 for each parameter under recency score, frequency score and monetary score. Please note that the rating of 5 is highest and denotes excellence of customer in relation to that parameter while 0 denotes lowest rating. The final RFM score is just the string type attachment of recency score and frequency score, and accordingly the customers are segmented into 10 types.')
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Scatter Plot', key = '13'):
        st.subheader('Scatter Plot')
        rfm.Scatter()
        st.write('Here you can Visualize the Distribution of Customer Depending on the Scatter Plot using different Features of Your choice. You can draw inference of the effect of one features on Other Features and see how Customers are distributed')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Bar Plot', key = '14'):
        st.subheader('Bar Plot')
        rfm.bar_plot()
        st.write('Above plot denotes the distribution of customers in different segments. Refer the details below about each customer segment and the corresponding remedy to increase your business:-')
        data = [['Champions','Customers that purchased recently, buy often and spend the most.','Continue enhancing their experience.'], ['Loyal customers','These customers are buy on a regular basis and are responsive to promotions.','Always be in touch and offer something extra'], ['Potential Loyalist','These are recent customers with average frequency.',' Connect with them and promote your business.  '],['Need Attention','The customers have above average recency, frequency and monetary values.','Try to provide attractive offers and special discounts to them.'],['At Risk','These customers purchase more number of times but with large time gaps.','Focus on making the customers more regular.'],['Can’t Lose','Customer that used to purchase frequently but haven’t returned for a long time.','Enquire about them and advertise about your business.'],['About To Sleep','These customers have below average recency and frequency values.','Keep reminding them about the quality you deliver.'],['New customers','They have bought most recently, but not often.','Offer them good service so that they revisit.'],['Promising','They are recent shoppers, but haven’t spent much.','Convince them to purchase to their full capability.'],['Hibernating','Their last purchase was long back and had a low number of orders','Try to contact them and persuade them to visit.']]
        df_rfm = pd.DataFrame(data, columns = ['Customer Segment', 'Details','Remedy to Bring Back'])
        st.table(df_rfm)
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
        
      if st.sidebar.checkbox('Treemap Plot', key = '15'):
        st.subheader('Treemap Plot')
        rfm.Tree_map()
        st.write('The above shown tree map gives a better understanding of the share of each of the customer segment. We hope that the segments of champions and loyal customers get or are maximized in your case')
      if st.sidebar.checkbox('Access Customer Info ', key = '16'):
        st.subheader('Access Customer Information')
        rfm.Access_Cust()
        st.write('Looks like this customer is closely associated to the growth of your business. Hope this customer fits into the top segments.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)

    if st.sidebar.checkbox('HYBRID_SEGMENTATION', key = '17'):
      st.sidebar.subheader('Choose Method for Cluster labelling')
      st.subheader('Under this section, the customer data you had fed previously will be segmented not just on the basis of RFM scores but also the clustering technique.')
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('SHOW DENDOGRAM', key = '18'):
        st.subheader('Dendogram Plot')
        st.write(h_rfm.Pre_hybrid())
        st.write('The above dendogram, having a tree like structure basically depicts relationship between all the data points of the input dataset. The different levels of dendogram each indicate clusters of data and how every small cluster gradually combines into a single cluster.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('SHOW ELBOW CURVE', key = '19'):
        st.subheader('Elbow Curve')
        st.write(h_rfm.elbow())
        st.write('In the above elbow graph, there are some sharp points that indicate minimum distortion. The total number of these sharp points denote the number of clusters that can segment the provided data efficiently')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Check Skewness', key = '20'):
        st.subheader('Skewness Information')
        st.write(h_rfm.skew_data())
        st.write('Above table contains the skewness and kurtosis values of the attributes required for hybrid segmentation. Both skewness and kurtosis help in analyzing the location and variability of a data set. Negative values for the skewness indicate data that are skewed left and positive values for the skewness indicate data that are skewed right. Positive kurtosis indicates a "heavy-tailed" distribution and negative kurtosis indicates a "light tailed" distribution. Take a look at your dataset before being transformed for hybrid segmentation.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if  st.sidebar.checkbox('Visualize Transformed Data', key = '21'):
        st.subheader('Visualizw Transformed Data for Hybrid Segmentation')
        h_rfm.Trans_data()
        st.write('Take a glimpse of the transformation that your original dataset undergo for a near to ideal skewness and kurtosis. Analyse the above graphs with respect to the origin and see its distribution. If still there is some skewness in the data, the final results can’t be trustworthy.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Transformed data', key = '22'):
        st.subheader(' Transformed Data for Hybrid Segmentation')
        st.write(h_rfm.Transform().astype('object'))
        st.write('The above table represents the transformed data that will be used for hybrid customer segmentation.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('KMEANS SEGMENTATION DataFrame', key = '23'):
        st.subheader('Kmeans Segmentation Dataframe')
        st.download_button(label="Download Kmeans Segmentation as CSV",data=h_rfm.KM().to_csv().encode('utf-8'),file_name='K-Means_Segmentation.csv',mime='text/csv',)
        st.write(h_rfm.KM())
        st.write('The above data table represents the segmented data using K-means and RFM score methods i.e. hybrid method. The ‘0’ value under cluster indicates customers with higher RFM or high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
	
   
      if st.sidebar.checkbox('Access Customer Info ', key = '168'):
        st.subheader('Access Customer Information')
        h_rfm.Access_hybrid()
        st.write('Looks like this customer is closely associated to the growth of your business. Hope this customer fits into the top segments.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('BOX PLOTS OF CLUSTERS', key = '24'):
        st.subheader('Box Plot of Clusters')
        h_rfm.BOX()
        st.write('The above plots indicate if your data is symmetrical, how tightly your data is grouped, and if and how your data is skewed')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('SCATTER PLOTS OF CLUSTERS' , key = '25'):
        st.subheader('Scatter Plot of Clusters')
        h_rfm.scatter_Cluster()
        st.write('The above plot indicates the clusters between Recency and Frequency or can compare monetary values with Recency and Frequency with the help of these scatter plots.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
        
      if st.sidebar.checkbox('Best Possible Number of Cluster', key = '26'):
        st.subheader('Best Possible number of Clusters ')
        st.write(h_rfm.Best_K())
        st.write('The above Result indicates the Best posible Value of Number of clusters possible for better Results,this is calculates using silhoutte score for each number of cluster and best value from those is choosen,Visualizing can be done using Dendogram plot and Elbow Curve ')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
  if choice  == 'Customer Classification':
    if st.sidebar.checkbox('Display Accuracy of all Classification models', key = '27'):
     st.subheader('Accuracy of all Classification models')
     ll.accuracy_ML()
     st.write('Since no single form of classification is appropriate for all datasets, we have  made a vast toolkit of off-the-shelf classifiers available as shown above.Refer the above accuracy scores and classify your dataset with the top three accurate models at least for better results.')
     add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
     st.markdown(add_line, unsafe_allow_html=True)

    if st.sidebar.checkbox('Choose Classifier for Prediction', key = '28'):
     st.sidebar.text('Choose Classifier for Prediction')
     ll.inp_classifier()
     add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
     st.markdown(add_line, unsafe_allow_html=True)

     if st.sidebar.button('Support Vector Machine', key = '33'):
        st.subheader('Prediction using Support Vector Machine')
        st.download_button(label=" Download SVM Predictions as csv",data=ll.support().to_csv().encode('utf-8'),file_name='SVM_Prediction.csv',mime='text/csv',)
        st.write(ll.support())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
     if st.sidebar.button('LogisticRegression', key = '30'):
        st.subheader('Prediction using Logistic Regression')
        st.download_button(label=" Download Logistic Regression as csv",data=ll.Logistic_reg().to_csv().encode('utf-8'),file_name='LogisticRegression_Prediction.csv',mime='text/csv',)
        st.write(ll.Logistic_reg())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
     if st.sidebar.button('k-Nearest Neighbors Algorithm', key = '31'):
        st.subheader('Prediction using k-Nearest Neighbors Algorithm')
        st.download_button(label=" Download  k-Nearest Neighbors prediction as csv",data=ll.KNN().to_csv().encode('utf-8'),file_name='k-Nearest Neighbors_Prediction.csv',mime='text/csv',)
        st.write(ll.KNN())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
     if st.sidebar.button('Decision Tree', key = '69'):
        st.subheader('Prediction using Decision Tree')
        st.download_button(label=" Download Decision Tree prediction as csv",data=ll.Dec_Tree().to_csv().encode('utf-8'),file_name=' Decision Tree_Prediction.csv',mime='text/csv',)
        st.write(ll.Dec_Tree())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
     if st.sidebar.button('Random Forest Model: ', key = '70'):
        st.subheader('Prediction using Random Forest Model')
        st.download_button(label=" Download Random Forest Model prediction as csv",data=ll.Random_F().to_csv().encode('utf-8'),file_name=' Random Forest Model_Prediction.csv',mime='text/csv',)
        st.write(ll.Random_F())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
     if st.sidebar.button('Ababoost Classifier Model: ', key = '71'):
        st.subheader('Prediction using Ababoost Classifier Model')
        st.download_button(label=" Download Ababoost Classifier Model prediction as csv",data=ll.Adaboost().to_csv().encode('utf-8'),file_name=' Ababoost Classifier Model_Prediction.csv',mime='text/csv',)
        st.write(ll.Adaboost())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
     if st.sidebar.button('Gradient Boosting Classifier Model: ', key = '72'):
        st.subheader('Prediction using Gradient Boosting Classifier Model')
        st.download_button(label=" Download Gradient Boosting Classifier Model prediction as csv",data=ll.Graddient().to_csv().encode('utf-8'),file_name=' Gradient Boosting Classifier_Prediction.csv',mime='text/csv',)
        st.write(ll.Graddient())
        st.write('Above dataframe represents the result for the classifier model selected. The ‘0’ value under cluster indicates customers with high valued customers while customers with ‘1’ value indicate comparatively less valued customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
  if choice == 'Customer Linked Predictions':
    st.sidebar.subheader('Here we will Explore Customer Lifetime Value')
    if st.sidebar.checkbox('CLTV Dataframe', key = '32'):
      st.download_button(label=" Download CLTV DataFrame as csv",data=cc.C_L_T_V().to_csv().encode('utf-8'),file_name='CLTV_DataFrame.csv',mime='text/csv',)
      st.write(cc.C_L_T_V())
      st.write('Watch out for the ‘Expected Average Profit’, ‘Customer lifetime value (CLV)’ and the segment that a particular customer falls under. Segment A indicates …, segment B indicates … , segment C indicates … and segment D indicates … ')
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)  
    if st.sidebar.checkbox('Aceess Customer Info'):
      cc.Access_cltv()
      st.write('Looks like this customer is closely associated to the growth of your business. Hope this customer fits into the top segments.')
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True) 
     
  if choice == 'Sales Forecasting':
    st.sidebar.subheader('Welcome to the world of forecasting!')
    st.subheader('Welcome to the section of sales forecasting!! This section will provide you the insights of your business performance in future by considering your present & past. So, get, set, forecast…')
    add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
    st.markdown(add_line, unsafe_allow_html=True)
    if st.sidebar.checkbox('30 Days Forecast'):
      st.subheader('30 Days Forecast')
      st.download_button(label="Download 30 Days Forecast as CSV",data=sf.LSTM_Model().to_csv().encode('utf-8'),file_name='30-Days_Forecast.csv',mime='text/csv',)
      st.write(sf.LSTM_Model())
      st.write('The above table represents the 30 day forecasted sales for your business. Take a look of how the customers will respond to your business in coming month!!')
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)
    if st.sidebar.checkbox('Date Filter', key = '31'):
     st.subheader('Access Forecasted information')
     sf.Access_Forecast()
     st.write('Looks like this particular date of the month is special for your business! Hope you are satisfied with the corresponding sales forecasting.')
     add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
     st.markdown(add_line, unsafe_allow_html=True)
  if choice == 'Forensic Analysis':
    
    if st.sidebar.checkbox('Cohort Analysis'):
      st.subheader('Let’s identify some groups within your dataset that share common characteristics and that can help your business generate a revenue in return.')
      st.sidebar.text('Choose among the below activities:')
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Monthly Revenue Plot'):
        st.subheader('Monthly Revenue Plot')
        chu.coh()
        st.write('The above line plot shows the generated revenue of your business over time. As from the plot, it can be referred that the time gap on an average is of a month, which indicates the monthly revenue trend of your business.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Montly Growth rate Plot'):
        st.subheader('Montly Growth rate Plot')
        chu.coh2()
        st.write('The above line plot shows the relative generated revenue of your business over time. If a particular season of months is considered, you can see how well your business had grew over time.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Montly Active Customers'):
        st.subheader('Montly Active Customers')
        st.download_button(label="Download Monthly Active Customer",data=chu.coh9().to_csv().encode('utf-8'),file_name='Monthly_Active_Customer.csv',mime='text/csv',)
        st.write(chu.coh9())
        st.write('The above table represents the number of total active customers for every month. These numbers can help you decide and plan your productions accordingly. A sharp increase in active customers indicates the peak season.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Montly Active Customers Plot'):
        st.subheader('Montly Active Customers Plot')
        chu.coh3()
        st.write('This bar plot gives the better visualisation of how the number of active customers change over time and what exact trend is going on.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Montly Total # of Orders'):
        st.subheader('Montly Total # of Orders')
        chu.coh4()
        st.write('This bar plot indicates the trend of purchases done by active customers. The trend in this plot can define a relationship between monthly active customers and the number of orders.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Monthly Order Average'):
        st.subheader('Monthly Order Average')
        chu.coh5()
        st.write('Monthly order average relates to the average revenue generated per order in that particular month of your business. Check out for the trend in the baove plot so that you can define relation between motnhly order average and active customers.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('New Usertype Vs Existing Usertype'):
        st.subheader('New Usertype Vs Existing Usertype')
        chu.coh6()
        st.write('New Customer Ratio is a good indicator, it shows if your business  is losing their existing customers or unable to attract new ones. In the above plot, check out for the nature of trend for new as well as existing customer. For ideal case, a less fluctuating upward trend of existing customers and a positive trend for new customers is appreciable for growth of your business.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('New Customer ratio'):
        st.subheader('New Customer ratio')
        chu.coh7()
        st.write('In the above plot, check out for the series of months or a particular month when new customers got influxed in your business. Watch for this kind of trend for over an year or two and devise your further product launches accordingly.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Montly Customer Retention '):
        st.subheader('Montly Customer Retention')
        st.download_button(label=" Download Monthly Customer Retention",data=chu.coh8().to_csv().encode('utf-8'),file_name='Monthly_Customer_Retention.csv',mime='text/csv',)
        st.write(chu.coh8())
        st.write('In the above table, the retention rate near to 1 is considered to be ideal while that of 0.5 can be appreciable for business growth. ')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Monthly retention Plot'):
        st.subheader('Monthly retention Plot')
        chu.coh_ret()
        st.write('The above line plot indicates the customer retention trend over the period of time.')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Monthly Retention Matrix'):
        st.subheader('Monthly Retention Matrix')
        st.write(chu.coh10())
        st.write('The values in each cell of above matrix represent if the customers were retained in that particular month or not. The value ‘1’ represents that most of the customers in previous month were retained in that particular month. ')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
      if st.sidebar.checkbox('Cohort based retention Matrix'):
        st.write(chu.coh11())
        st.write('The values in each cell of above matrix depict the relative rate of retention of a particular cohort of customers with respect to the previous month. The retnetion rate closer to 0.5 denotes appreciable growth. ')
        add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
        st.markdown(add_line, unsafe_allow_html=True)
  if choice == 'About':
    st.sidebar.title('About:')
    if st.sidebar.checkbox("See User guide"):
      st.subheader('User Guide: ')
      aa.User()
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)
    if st.sidebar.checkbox('MEET THE TEAM'):
      aa.Dev()
      add_line= '<p style="font-family:sans-serif; font-weight:bold;color:blue;font-size: 60px;">___________________________________________________</p>'
      st.markdown(add_line, unsafe_allow_html=True)
    if st.sidebar.checkbox('User Feedback'):
      aa.Feed()

if __name__=='__main__':
  dataframe = EDA_Analysis()
  rules = Product_Recommendation()
  rfm  = RFM_Analysis()
  h_rfm = Hybrid_Analysis()
  ll = ML_models()
  cc = cltv()
  sf = Sales_Forecasting()
  chu = Churn_Analysis()
  g = ghar()
  aa  = About()
  main()   
