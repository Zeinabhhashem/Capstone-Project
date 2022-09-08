
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import pandas as pd
import plotly as plt
import plotly.graph_objects as go
import plotly.express as px
import io 
import streamlit.components.v1 as components
import time
import os 
import cv2
from PIL import Image
from pathlib import Path
from IPython.display import HTML
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import datetime as dt
#import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import xgboost as xgb
   
filename = 'C:/Users/zeina/OneDrive/Desktop/streamproj'


st.set_page_config(
    page_title="Creative Ads Dashboard",
    page_icon="📱",
    layout="wide",
)
#st.markdown("""<"style> st.menu .st.nav-link {color: "#FFFFFF";}</style>""",unsafe_allow_html=True,)

with st.sidebar:
    choose = option_menu("App Menu", ["Upload Data", "Extraction", "Dashboard", "Model"],
                         icons=['house','eye', 'kanban', 'journal-bookmark-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#00072B"},
        "menu-icon": {"color": "#62DBDF"},
        "menu-title": {"color": "#62DBDF"},
        "icon": {"color": "#62DBDF", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px","color":"#FFFFFF","--hover-color": "#2B314C"},
        "nav-link-selected": {"background-color": "#2B314C"},
        
    }
    )
   

if choose == "Upload Data":
    st.title("**Analyzing Creative Ads**")
    img = Image.open("Digital-ads3.png")
    #".progress-bar{background-color:#03bb85;}"
    # st.markdown("""<style>st.Progress .st.bo{
    # background-color: "#62DBDF";}</style>""", unsafe_allow_html=True)
    col1, col2 = st.columns( [0.4, 0.4])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {font-size:45px';} # </style> """, unsafe_allow_html=True)
        st.header('**Upload the Data**')
        #st.subheader('<p class="font"><strong>Upload the Data<strong></p>', unsafe_allow_html=True)
        st.write('<p style="font-size: 20px;">This project has been introduced into a pipeline framework to analyze creative ads. To make the process easier this app has been developed to automate all the pre-processing steps, feature extraction from images and predictive model creation. This tool can also be used to address other clients’ needs that fall within the umbrella of creative ad performance.', unsafe_allow_html=True)    
    col1, col2 = st.columns([0.15, 0.3])
    with col1:  
        st.markdown("""<style>.st-bn {background-color: rgb(98,219,223);}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.css-1den1ap {background-color: #62DBDF;}</style>""", unsafe_allow_html=True)
        uploaded_data = st.file_uploader('Upload dataset', type='csv')
    if uploaded_data is not None:
        data = pd.read_csv(uploaded_data)
        data= data.drop(labels=['Ad_Preview_Shareable_Link','Ad_Creative_Image_URL', 'Website_Conversions','Adds_to_Cart'], axis=1)
        if st.checkbox('Show Dataset for Creative Ads'):
          data = data
          data
    st.image(img, width=1800)
    
    
if choose == "Extraction":
    st.header("**Creative Ads Fearure Extraction**")
    st.markdown("""<style>.st-bo {background-color: rgb(98,219,223);}</style>""", unsafe_allow_html=True)
    st.markdown("""<style>.st-bn {background-color: rgb(98,219,223);}</style>""", unsafe_allow_html=True)
    # st.markdown("""<style>.css-17lpgg4 {background-color: #62DBDF;}</style>""", unsafe_allow_html=True)
    # st.markdown("""<style>.css-sop4c {code-color: rgb(98,219,223);}</style>""", unsafe_allow_html=True)
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')
    col1, col2 = st.columns([0.3, 0.8])
    with col1:
      # Set up tkinter
      root = tk.Tk()
      root.withdraw()

      # Make folder picker dialog appear on top of other windows
      root.wm_attributes('-topmost', 1)
      #root.mainloop()
      # Folder picker button
      st.subheader('Please select the folder with the images:')
      clicked = st.button('Upload Image Folder here')
      latest_iteration2 = st.empty()
      if clicked:
        dirname = st.text_input('Selected folder:', filedialog.askdirectory(master=root))
        #root.mainloop()
        #data = pd.read_csv(uploaded_data)
        #
        # The gender model architecture
        # https://drive.google.com/open?id=1W_moLzMlGiELyPxWiYQJ9KFaXroQ_NFQ
        GENDER_MODEL = 'models/deploy_gender2.prototxt'
        # The gender model pre-trained weights
        # https://drive.google.com/open?id=1AW3WduLk1haTVAxHOkVS_BEzel1WXQHP
        GENDER_PROTO = 'models/gender_net.caffemodel'
        # Each Caffe Model impose the shape of the input image also image preprocessing is required like mean
        # substraction to eliminate the effect of illunination changes
        MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        # Represent the gender classes
        GENDER_LIST = ['Male', 'Female']
        # https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
        FACE_PROTO = "models/deploy.prototxt.txt"
        # https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
        FACE_MODEL = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"


        # load face Caffe model
        face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
        # Load gender prediction model
        gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

        @st.cache(suppress_st_warning=True)
        def get_faces(frame, confidence_threshold=0.5):
            # convert the frame into a blob to be ready for NN input
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177.0, 123.0))
            # set the image as input to the NN
            face_net.setInput(blob)
            # perform inference and get predictions
            output = np.squeeze(face_net.forward())
            # initialize the result list
            faces = []
            number_faces = 0
            # Loop over the faces detected
            for i in range(output.shape[0]):
                confidence = output[i, 2]
                if confidence > confidence_threshold:
                    box = output[i, 3:7] * \
                        np.array([frame.shape[1], frame.shape[0],
                                frame.shape[1], frame.shape[0]])
                    number_faces = number_faces + 1
                    # convert to integers
                    start_x, start_y, end_x, end_y = box.astype(int)
                    # widen the box a little
                    start_x, start_y, end_x, end_y = start_x - 10, start_y - 10, end_x + 10, end_y + 10
                    start_x = 0 if start_x < 0 else start_x
                    start_y = 0 if start_y < 0 else start_y
                    end_x = 0 if end_x < 0 else end_x
                    end_y = 0 if end_y < 0 else end_y
                    #area = ((end_x - start_x)*(end_y - start_y)) + area
                    # append to our list
                
                    faces.append((start_x, start_y, end_x, end_y))
        
                return faces, number_faces


        @st.cache(suppress_st_warning=True)
        def display_img(title, img):
            """Displays an image on screen and maintains the output until the user presses a key"""
            # Display Image on screen
            cv2.imshow(title, img)
            # Mantain output until user presses a key
            cv2.waitKey(0)
            # Destroy windows when user presses a key
            cv2.destroyAllWindows()

        @st.cache(suppress_st_warning=True)
        def get_optimal_font_scale(text, width):
            """Determine the optimal font scale based on the hosting frame width"""
            for scale in reversed(range(0, 60, 1)):
                textSize = cv2.getTextSize(text, fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=scale/10, thickness=1)
                new_width = textSize[0][0]
                if (new_width <= width):
                    return scale/10
            return 1
        @st.cache(suppress_st_warning=True)
        def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
            # initialize the dimensions of the image to be resized and
            # grab the image size
            dim = None
            (h, w) = image.shape[:2]
            # if both the width and height are None, then return the
            # original image
            if width is None and height is None:
                return image
            # check to see if the width is None
            if width is None:
                # calculate the ratio of the height and construct the
                # dimensions
                r = height / float(h)
                dim = (int(w * r), height)
            # otherwise, the height is None
            else:
                # calculate the ratio of the width and construct the
                # dimensions
                r = width / float(w)
                dim = (width, int(h * r))
            # resize the image
            return cv2.resize(image, dim, interpolation = inter)
        @st.cache(suppress_st_warning=True)
        def predict_gender(directory):
            df = {}
            j = 0
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                latest_iteration2.text("""Predict the gender of the faces showing in the images""")
                # Read Input Image
                img = cv2.imread(directory + "/" + filename)
                # resize the image, uncomment if you want to resize the image
                # img = cv2.resize(img, (frame_width, frame_height))
                # Take a copy of the initial image and resize it
                frame = img.copy()
                h, w = img.shape[:2]
                areaframe = h*w
                # if frame.shape[1] > frame_width:
                #     frame = image_resize(frame, width= frame_width)
                # predict the faces
                faces, number_faces = get_faces(frame)
                # Loop over the faces detected
                # for idx, face in enumerate(faces):
                for i, (start_x, start_y, end_x, end_y) in enumerate(faces):
                    face_img = frame[start_y: end_y, start_x: end_x]
                    blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False, crop=False)
                    # Predict Gender
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    i = gender_preds[0].argmax()
                    gender = GENDER_LIST[i]
                    gender_confidence_score = gender_preds[0][i]
                    # Draw the box
                    label = "{}-{:.2f}%".format(gender, gender_confidence_score*100)
                    #print(label)
                    yPos = start_y - 15
                    while yPos < 15:
                        yPos += 15
                    area = ((end_x - start_x)*(end_y - start_y))
                    prop = area/areaframe
                    # get the font scale for this image size
                    optimal_font_scale = get_optimal_font_scale(label,((end_x-start_x)+25))
                    box_color = (255, 0, 0) if gender == "Male" else (147, 20, 255)
                    cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), box_color, 2)
                    # Label processed image
                    cv2.putText(frame, label, (start_x, yPos),
                                cv2.FONT_HERSHEY_SIMPLEX, optimal_font_scale, box_color, 2)

                    print(filename , ":", label , ":", prop, ":", number_faces)
                    df[j] = [filename, label, number_faces, prop, area, areaframe]
                    j = j + 1
                    cv2.imwrite('detectedfaces/'+ file + f'_detected.jpg', frame)
                    cv2.destroyAllWindows()

            df = pd.DataFrame.from_dict(df, orient='index', columns=['Creative_ID','Gender','Number_of_Faces','Prop','Area','Areaframe'])
            
                      

            df.to_csv('genderdetectionoutput.csv', index=False)
            return df
                    

        predict_gender(dirname)
    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        df = pd.read_csv('genderdetectionoutput.csv')
        if st.checkbox('Show Dataset for Creative Ads'):
            #clean = HTML(data.to_html(index=False))
            df = df
            df['Creative_ID'] = df['Creative_ID'].str.replace(r'.jpg', '').astype('int64')
            df['Total Proportion'] = df.groupby(df['Creative_ID'].ne(df['Creative_ID'].shift()).cumsum())['Prop'].transform('sum')
            df['Total Area'] = df.groupby(df['Creative_ID'].ne(df['Creative_ID'].shift()).cumsum())['Area'].transform('sum')
            df['Gender_Stripped']= df["Gender"].apply(lambda x: x.split("-")[0])

            df['Count'] = ''
            df['Genders'] = df.groupby(df['Creative_ID'].ne(df['Creative_ID'].shift()).cumsum())['Gender_Stripped'].transform('count')
            df3 = df[['Creative_ID', 'Gender_Stripped', 'Count']].groupby(['Creative_ID', 'Gender_Stripped']).count().reset_index()
            df3['Final_Gender'] = ''
            for row in df3['Creative_ID'].unique():
                tempdf = df3[df3['Creative_ID'] == row]
                if tempdf.shape[0] == 2:
                    df3['Final_Gender'][df3['Creative_ID'] == row]='Both'
                else:
                    df3['Final_Gender'][df3['Creative_ID'] == row] = df3['Gender_Stripped'][df3['Creative_ID'] == row]
            df3 = df3.drop(labels='Gender_Stripped', axis=1)
            df3.drop_duplicates(inplace=True)
            df = df.drop(labels=['Gender', 'Prop', 'Area', 'Gender_Stripped', 'Count', 'Genders'], axis=1)
            df.drop_duplicates(inplace=True)
            df = pd.merge(df, df3, on ='Creative_ID')
            df
            

    if uploaded_data is not None:
        df = pd.read_csv('genderdetectionoutput.csv')
        df['Creative_ID'] = df['Creative_ID'].str.replace(r'.jpg', '').astype('int64')
        df['Total Proportion'] = df.groupby(df['Creative_ID'].ne(df['Creative_ID'].shift()).cumsum())['Prop'].transform('sum')
        df['Total Area'] = df.groupby(df['Creative_ID'].ne(df['Creative_ID'].shift()).cumsum())['Area'].transform('sum')
        df['Gender_Stripped']= df["Gender"].apply(lambda x: x.split("-")[0])
        df['Count'] = ''
        df['Genders'] = df.groupby(df['Creative_ID'].ne(df['Creative_ID'].shift()).cumsum())['Gender_Stripped'].transform('count')
        df3 = df[['Creative_ID', 'Gender_Stripped', 'Count']].groupby(['Creative_ID', 'Gender_Stripped']).count().reset_index()
        df3['Final_Gender'] = ''
        for row in df3['Creative_ID'].unique():
            tempdf = df3[df3['Creative_ID'] == row]
            if tempdf.shape[0] == 2:
                df3['Final_Gender'][df3['Creative_ID'] == row]='Both'
            else:
                df3['Final_Gender'][df3['Creative_ID'] == row] = df3['Gender_Stripped'][df3['Creative_ID'] == row]
        df3 = df3.drop(labels='Gender_Stripped', axis=1)
        df3.drop_duplicates(inplace=True)
        df = df.drop(labels=['Gender', 'Prop', 'Area', 'Gender_Stripped', 'Count', 'Genders'], axis=1)
        df.drop_duplicates(inplace=True)
        df = pd.merge(df, df3, on ='Creative_ID')
    
        data = pd.read_csv(uploaded_data)
        data= data.drop(labels=['Ad_Preview_Shareable_Link','Ad_Creative_Image_URL', 'Website_Conversions','Adds_to_Cart'], axis=1)
        data.dropna(inplace=True)
        data["Language"] = data["Campaign_Name"].map(lambda x: "Arabic" if "AR" in x else "English" if "EN" in x else "English|Arabic")
        data['Country']= data["Campaign_Name"].apply(lambda x: x.split("_")[1])
        data['Campaign']= data["Campaign_Name"].apply(lambda x: x.split("_")[2])
        data= data.drop(labels=['Data_Source_description', 'AllSource', 'Ad_ID'], axis=1)
        distinctdates = data.groupby('Creative_ID')['Date'].nunique().reset_index(name='number_of_distinct_dates')
        data = pd.merge(data, distinctdates, on ='Creative_ID')
        data = data.drop(labels=['Date', 'Campaign_Name'], axis=1)
        data['Total Impressions'] = data.groupby(data['Creative_ID'].ne(data['Creative_ID'].shift()).cumsum())['Impressions'].transform('sum')
        data['Total Clicks'] = data.groupby(data['Creative_ID'].ne(data['Creative_ID'].shift()).cumsum())['Clicks__all_'].transform('sum')
        data['Total Amount Spent'] = data.groupby(data['Creative_ID'].ne(data['Creative_ID'].shift()).cumsum())['Amount_Spent__USD_'].transform('sum')
        data['Total Link Clicks'] = data.groupby(data['Creative_ID'].ne(data['Creative_ID'].shift()).cumsum())['Link_Clicks'].transform('sum')
        data= data.drop(labels=['Impressions', 'Clicks__all_', 'Amount_Spent__USD_', 'Link_Clicks','Video_Watches_at_25_','Video_Watches_at_50_','Video_Watches_at_75_','Video_Watches_at_100_'], axis=1)
        data.drop_duplicates(inplace=True)
        #convert year variable in df2 to integer
        data['Creative_ID']=data['Creative_ID'].astype('int64')
        df['Creative_ID']=df['Creative_ID'].astype('int64')
        data = data.merge(df, on='Creative_ID', how='left')
        data.dropna(inplace=True)
        data.drop_duplicates(inplace=True)
        data['CTR'] = (data['Total Clicks']/data['Total Impressions'])*100
        def cat(x):
            if x <0.5:
                return "Low"
            elif 1 > x > 0.5:
                return "Medium"
            else:
                return "High"

        for col in data.columns:
            data['Category'] = data['CTR'].apply(lambda x: cat(x))
        data['Total_Area_Cat'] = pd.qcut(data['Total Area'], q=3, labels=['SMALL','Medium','LARGE'])
        data.to_csv('Dataagg.csv', index=False)
        if st.checkbox('Show Combined Dataset of Total Metrics and Features'):
            data = data
            data

  
if choose == "Dashboard":
    st.markdown("""<style>.css-17lpgg4 {background-color: #dedcd7;}</style>""", unsafe_allow_html=True)
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')
    data = pd.read_csv('Dataagg.csv')
    st.header("**Creative Ads Dashboard**")
    
    if uploaded_data is not None:
      data = data
      st.markdown("""<style>.st-bp {background-color: rgb(0,7,43);}</style>""", unsafe_allow_html=True)
      #st.markdown("""<style>.st-bn {background-color: rgb(0,7,43);}</style>""", unsafe_allow_html=True)
      my_bar = st.progress(0)
      st.markdown("""<style>.css-15u18b5 {font-size: 18px;}</style>""", unsafe_allow_html=True)
      for percent_complete in range(100):
          time.sleep(0.1)
          my_bar.progress(percent_complete + 1) 
      try:
        col1, col2, col3 = st.columns([0.2, 0.2 ,0.2])
        with col1:
            country_names = {'EGY': 'Egypt','BHR': 'Bahrain','KWT': 'Kuwait','QAT': 'Qatar','KSA': 'Saudi Arabia','JOR': 'Jordan','OMN': 'Oman','GCC': 'GCC','PS': 'Palestine','IRQ': 'Iraq','ALG': 'Algeria','TUN': 'Tunisia','CHAD': 'Chad','UAE': 'United Arab Emirates'}
            data['country_full'] = data['Country'].map(country_names)
            country = st.multiselect("Select The Country", options=data['country_full'].unique())
            if country:
              with col2:
                campaign = st.multiselect("Select the Country",options=data[data["country_full"].isin(country)]["Campaign"].unique())
            else:
              with col2:
                campaign = st.multiselect("Select the Campaign",options=data['Campaign'].unique())

            if campaign:
              with col3:
                CTR = st.multiselect("Select the CTR Category",options=data[data["Campaign"].isin(campaign)]["Category"].unique())
            elif country:
              with col3:
                CTR = st.multiselect("Select the CTR Category",options=data[data["country_full"].isin(country)]["Category"].unique())
            else :
              with col3:
                CTR = st.multiselect("Select the CTR Category",options=data["Category"].unique())
    
      except Exception as e:
          pass

      finally:

            if country:
              data = data.query("country_full in @country")    
        
            if campaign:
              data = data.query("Campaign in @campaign")

            if CTR:
              df2 = data.query("Category in @CTR")
            # create three columns
      def human_format(num):
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

      Amount_Spent = data['Total Amount Spent'].sum()
      Impressions = data['Total Impressions'].sum()
      Clicks = data['Total Clicks'].sum()
      Link_Clicks = data['Total Link Clicks'].sum()
      Number_of_Faces = data['Number_of_Faces'].sum()
      Campaigns =  len(data["Campaign"].value_counts())


      CTR = data['CTR'].sum()
      

      kpi1, kpi2, kpi3 = st.columns(3)

       # fill in those three columns with respective metrics or KPIs
      
      st.markdown("""<style>.css-1gdrjus {font-size: 25px;}</style>""", unsafe_allow_html=True)
      st.markdown("""<style>.css-1m4nakq {font-size: 18px;}</style>""", unsafe_allow_html=True)
      kpi1.metric(label="Total Amount Spent 💵 ",value=human_format(Amount_Spent))

      kpi2.metric(label="Total Number of Impressions👀",value=human_format(Impressions))

      kpi3.metric(label="Total Clicks↗️",value= human_format(Clicks))

      kpi4, kpi5, kpi6 = st.columns(3)
      kpi4.metric(label="Total Link Clicks🔗",value=human_format(Link_Clicks))

      kpi5.metric(label="Total Number of Faces👤",value=human_format(Number_of_Faces))

      kpi6.metric(label="Total Campaigns📄",value=human_format(Campaigns))
      
      col1, col2, col3, col4 = st.columns( [0.3,0.4, 0.3, 0.4])
      with col2:
        st.markdown("**Top 5 Campaigns**")
      with col4:
        st.markdown("**Campaign Distribution by CTR**")

      col1, col2, col3, col4= st.columns([0.1,0.1,0.5,0.8])
      img2 = Image.open("Male.png")
      img3 = Image.open("Female.png")
      space = Image.open("space.png")
      female = data.Final_Gender.value_counts().Female
      male = data.Final_Gender.value_counts().Male
      with col1:
        st.image(space, width=40)
      with col1:
        st.image(img2, width=100)
      with col2:
        st.image(space, width=40)
      with col2:
        col2.metric(label="Male",value=human_format(male))
      with col1:
        st.image(img3, width=120)
      with col2:
        st.image(space, width=40)
      with col2:
        col2.metric(label="Female",value=human_format(female)) 


      topitems = data.groupby("Campaign").sum()["CTR"].sort_values(ascending=False)
      topitems = topitems.reset_index()
      top5items = topitems.head(5)
      pie1 = px.pie(top5items, values='CTR', names='Campaign', color_discrete_sequence=px.colors.sequential.Purp,hover_data=['Campaign'], labels = {'CTR'},hole=.3)
      pie1.update_traces(textposition='inside', textinfo='percent')
      with col3:
        st.plotly_chart(pie1, use_container_width=False)
      CTR_by_country=data.groupby("country_full")["CTR"].sum().sort_values(ascending=True)


      CTR_by_country = CTR_by_country.reset_index()

                 #data["Count"] = data.iloc[:,1]
      map = px.choropleth(CTR_by_country, locations="country_full",color="CTR",hover_name="country_full",color_continuous_scale=px.colors.sequential.dense, locationmode="country names")
      map.update_layout(margin=dict(l=0, r=0, t=0, b=0),geo = dict(showframe = False,showcoastlines=False, projection={'type':'equirectangular'}))
      #fig.update_traces(geo_bgcolor="#F7F3EB")
      map.update_geos(fitbounds="locations", visible=True)
      map.update_layout({'geo_bgcolor': 'rgb(247, 243, 235)'})
      with col4:
        st.plotly_chart(map, use_container_width=True)
      topCTRDates = data.groupby("number_of_distinct_dates").mean()["CTR"].sort_values(ascending=False)
      topCTRDates = topCTRDates.reset_index()
      topCTRSpent = data.groupby("Total Amount Spent").mean()["CTR"].sort_values(ascending=False)
      topCTRSpent = topCTRSpent.reset_index()
      bar = px.bar(topCTRDates, x='number_of_distinct_dates', y='CTR', color='CTR',color_continuous_scale=px.colors.sequential.dense )#color='Category',color_discrete_map={
      bar.update_layout({'plot_bgcolor':'rgb(247, 243, 235)','paper_bgcolor':'rgb(247, 243, 235)'})
      fig2 = px.scatter(data, x="Total Amount Spent", y="CTR", color="CTR",color_continuous_scale=px.colors.sequential.dense,size='Total Amount Spent', hover_data=['Total Amount Spent'])
      fig2.update_layout({'plot_bgcolor':'rgb(247, 243, 235)','paper_bgcolor':'rgb(247, 243, 235)'})
      #fig2.update_traces(color=['#62DBDF', '#9473CC','#00072B'], showlegend=False)
      col1, col2, col3, col4= st.columns( [0.3,0.3,0.5,0.6])
      with col2:
        st.markdown('CTR Over Number of Dates per Creative Ad')
      with col4:
        st.markdown('CTR Over Total Amount Spent per Creative Ad')
      col1, col2 = st.columns([0.5,0.5])
      with col1:
        st.plotly_chart(bar, use_container_width=True)
      with col2: 
        st.plotly_chart(fig2, use_container_width=True)
  
agg = 'Dataagg.csv'
if choose == "Model":
    st.header("**CTR Prediction**")
    st.subheader("How will the Creative Ad perform?")
    st.markdown("""<style>.css-17lpgg4 {background-color: rgb(222 220 215);}</style>""", unsafe_allow_html=True) 
    uploaded_data = st.sidebar.file_uploader('Upload dataset', type='csv')

    if uploaded_data is not None:
      agg = pd.read_csv(agg)
      col = ['number_of_distinct_dates','Total Amount Spent','Number_of_Faces','Total_Area_Cat','Final_Gender','Campaign','Country']
      X_dummy =  agg[col]
      y_dummy = agg['Category']
      dic = {'High': 2, 'Medium':1, 'Low':0}
      # X =  agg[col]
      # y = agg['Category'].values
      # X =(pd.get_dummies(X)).values
      y_dummy = y_dummy.map(dic)
      X_dummy =pd.get_dummies(X_dummy).values
      xgboost = xgb.XGBClassifier(random_state=0,use_label_encoder=False)
      #decision_tree = DecisionTreeClassifier(random_state=0)

      #predictions = xgboost.fit(X_dummy, y_dummy).predict(X_dummy)
      cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
      #cv = RepeatedStratifiedKFold(n_splits=5, random_state=1, n_repeats =5)
      #n_scores = cross_val_score(decision_tree, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
      
      # n_scoresxg = cross_val_score(xgboost, X_dummy, y_dummy, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
      # #prediction = "AUC: %0.2f (+/- %0.2f)" % (n_scores.mean(), n_scores.std() * 2)
      # prediction = "AUC: %0.2f (+/- %0.2f)" % (n_scoresxg.mean()*100,n_scoresxg.std() * 2 *100)
      #def thousand_sep(num):
          #return ("{:,}".format(num))
      #adding a button
      from stqdm import stqdm

      if st.button('Train the Model'):
        st.markdown("""<style>.st-bp {background-color: rgb(98, 219, 223);}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.st-bn {background-color: rgb(98, 219, 223);}</style>""", unsafe_allow_html=True)
        latest_iteration = st.empty()
        my_bar = st.progress(0)
        st.markdown("""<style>.css-15u18b5 {font-size: 18px;}</style>""", unsafe_allow_html=True)
        # for _ in stqdm(range(50)):
        #   for _ in stqdm(range(15)):
        #      sleep(0.5)
        for percent_complete in range(100):
          latest_iteration.text(f'Model is Training...')
          time.sleep(0.1)
          my_bar.progress(percent_complete + 1)
      
        predictions = xgboost.fit(X_dummy, y_dummy).predict(X_dummy)
        n_scoresxg = cross_val_score(xgboost, X_dummy, y_dummy, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        prediction = "AUC: %0.2f (+/- %0.2f)" % (n_scoresxg.mean()*100,n_scoresxg.std() * 2 *100)
        st.markdown("""<style>.st-h8 {background-color: rgb(189, 215, 216);}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.st-h7 {color: rgb(0,0,128);}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.st-fr {background-color: rgb(189, 215, 216);}</style>""", unsafe_allow_html=True)
        st.markdown("""<style>.st-fq {color: rgb(0,0,128);}</style>""", unsafe_allow_html=True)
        #st.markdown("""<style>.css-sop4c {font-family: "Source Code Pro", sans-serif;}</style>""", unsafe_allow_html=True)
        st.success("Training Complete!")
        
        st.write('<p style="font-size: 25px;">This model can predict on average ',prediction,  '% of the creative ads.', unsafe_allow_html=True)

        accuracy = n_scoresxg.mean()*100
        if accuracy> 70:
          st.write('<p style="font-size: 25px;">More data is needed to increase model accuracy.', unsafe_allow_html=True)
        else:
          st.write('<p style="font-size: 25px;">Model requires more data to increase accuracy.', unsafe_allow_html=True)

    
  
      
       
     

   
    
    

      


   
      
     

  
          
           
     