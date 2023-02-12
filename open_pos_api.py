import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#from openpyxl import Workbook, load_workbook
#from openpyxl.utils import get_column_letter
from datetime import datetime, timedelta, date
import plotly.graph_objects as go
import plotly.express as px
import warnings
import streamlit as st
warnings.filterwarnings('ignore')


def home():
    st.title('OPEN POs - RECEIVED POs - POs WITH ETA: ANALYSIS')
    st.markdown('### Data Source: SAP - GOLD')

csv_file = st.file_uploader('Upload CSV File', type=['csv'])

if csv_file is not None:
    Gold_data = pd.read_csv(r'C:\Users\USUARIO\Downloads\Gold Data.csv', index_col=0, encoding='latin-1', parse_dates=True)   
    Gold_data.reset_index(inplace=True)
    Gold_data = Gold_data[['Field Purchase Order','Field PO Item','Material Number','Total Price (Value)','Field PO Vendor','HUB SO Process Class','Field PO Date','Field PO Request Date(RDD)','OBS Creation Date HUB','ETA','Field PO GR Date']]
    Gold_data['Field PO Date'] = pd.to_datetime(Gold_data['Field PO Date']).values
    Gold_data['Field PO Request Date(RDD)'] = pd.to_datetime(Gold_data['Field PO Request Date(RDD)']).values
    Gold_data['OBS Creation Date HUB'] = pd.to_datetime(Gold_data['OBS Creation Date HUB']).values
    Gold_data['Field PO GR Date'] = pd.to_datetime(Gold_data['Field PO GR Date']).values
    Gold_data[['Field Purchase Order','Field PO Item']] = Gold_data[['Field Purchase Order','Field PO Item']].astype(str)

def describe(df):
    st.markdown('### Data Statistics')
    st.write(df[df['Total Price (Value)'].notnull()])

def gr_pos(df):
    st.header('POs with GOOD RECEIVE')
    st.markdown('#### Number of days from PO creation to Good Receive by Process Class')
    Gold_GRs = df[df['Field PO GR Date'].notnull()]
    Gold_GRs.dropna(axis=0, inplace=True)
    Gold_GRs['hub_time_d'] = Gold_GRs['OBS Creation Date HUB'] - Gold_GRs['Field PO Date']
    Gold_GRs['lead_time_d'] = Gold_GRs['Field PO GR Date'] - Gold_GRs['Field PO Date']
    Gold_GRs['hub_time_d'] = Gold_GRs.hub_time_d.dt.days
    Gold_GRs['lead_time_d'] = Gold_GRs.lead_time_d.dt.days
    
    plt.style.use('fivethirtyeight')
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(10,5))
    ax1.pie(Gold_GRs.groupby(by=['HUB SO Process Class']).mean()['hub_time_d'], autopct='%.2f')
    ax2.pie(Gold_GRs.groupby(by=['HUB SO Process Class']).mean()['lead_time_d'], autopct='%.2f')
    ax1.set_ylabel('HUB Time Days', fontsize=14)
    ax2.set_ylabel('Lead Time Days', fontsize=14)
    fig.legend(Gold_GRs['HUB SO Process Class'].unique())
    fig.suptitle('MEAN TIME DAYS BS vs. BO')
    st.pyplot(fig)

    st.markdown('#### Average days by vendor to arrive to the hub and the final destination')
    x = Gold_GRs['Field PO Vendor'].unique()
    plt.style.use('fivethirtyeight')
    fig_1, ax = plt.subplots(figsize=(15,6))
    ax.bar(x, Gold_GRs.groupby(by='Field PO Vendor').mean()['lead_time_d'], label='Total Lead Time')
    ax.bar(x, Gold_GRs.groupby(by='Field PO Vendor').mean()['hub_time_d'], label='Days to HUB')
    plt.xlabel("Vendors")
    plt.ylabel("Avg_Days")
    plt.xticks(rotation=90)
    ax.legend()
    st.pyplot(fig_1)

def no_eta_pos(df):
    st.header('POs without ETA')
    st.markdown('#### Number of POs and average days from creation to current day by vendor')

    POs_no_ETA = df.drop(['OBS Creation Date HUB', 'Field PO Request Date(RDD)'], axis=1)
    POs_no_ETA = POs_no_ETA[POs_no_ETA['Field PO GR Date'].isna()]
    POs_no_ETA = POs_no_ETA[POs_no_ETA['ETA'].isna()]
    POs_no_ETA['current_date'] = pd.to_datetime(date.today())
    POs_no_ETA.drop(['ETA', 'Field PO GR Date'], axis=1, inplace=True)
    POs_no_ETA['elapsed_time'] = POs_no_ETA.current_date - POs_no_ETA['Field PO Date']
    POs_no_ETA['elap_time_days'] = POs_no_ETA.elapsed_time.dt.days
    POs_no_ETA = POs_no_ETA[POs_no_ETA['Field PO Date'] > '2022-01-01']
    
    values = round(POs_no_ETA.groupby(by=['Field PO Vendor']).mean()['elap_time_days'],2)
    df = POs_no_ETA.groupby(by=['Field PO Vendor']).count().reset_index()
    df = df.rename({'Field Purchase Order':'numPOs'}, axis=1)

    fig = px.treemap(df, path=[px.Constant("All POs"), 'Field PO Vendor',round(POs_no_ETA.groupby(by=['Field PO Vendor']).count()['elap_time_days'],2)], 
                    values=values, color_continuous_scale='RdBu', color='numPOs')
    fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    fig.data[0].textinfo='label+value'
    st.plotly_chart(fig)

    st.markdown('#### No ETA open orders by day ranges and vendor')
    delayed_POs = 180
    fig = px.histogram(POs_no_ETA[POs_no_ETA.elap_time_days > delayed_POs], x='elap_time_days', color='Field PO Vendor', nbins=16, text_auto=True)
    fig.update_traces(opacity=0.9)
    fig.update_layout(xaxis_title_text='Elapsed_days', yaxis_title_text='Number_POs', bargap=0.08)
    st.plotly_chart(fig)

def eta_pos(df):
    st.header('POs with ETA')
    st.markdown('#### Days distribution from POs creation to ETA by vendor')

    ETA_POs = Gold_data.drop(['Field PO Request Date(RDD)','OBS Creation Date HUB', 'Field PO GR Date'], axis=1)
    ETA_POs = ETA_POs[(ETA_POs.ETA.notna())]
    ETA_POs = ETA_POs[ETA_POs['Field PO Date']> '2022-01-01']
    ETA_POs = ETA_POs[~(ETA_POs['ETA'] == datetime.strptime('00:00:00', '%H:%M:%S').time())]
    ETA_POs = ETA_POs[~(ETA_POs.ETA == '0-Jan-00')]
    ETA_POs.ETA = pd.to_datetime(ETA_POs.ETA)
    ETA_POs['time_ETA'] = ETA_POs.ETA - ETA_POs['Field PO Date']
    ETA_POs.time_ETA = ETA_POs.time_ETA.dt.days
    delayed_POs = 180
    ETA_POs =  ETA_POs[ETA_POs.time_ETA > delayed_POs]

    sns.set_style('darkgrid')
    fig = plt.figure(figsize=(12,8))
    sns.boxplot(y=ETA_POs['Field PO Vendor'], x=ETA_POs.time_ETA, linewidth=1, saturation=5)
    plt.ylabel("Vendors")
    plt.xlabel("Days to ETA")
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(fig)

def tracking_pos(df):
    st.markdown('### Tracking Open POs')

    POs_no_ETA = df.drop(['OBS Creation Date HUB', 'Field PO Request Date(RDD)'], axis=1)
    POs_no_ETA = POs_no_ETA[POs_no_ETA['Field PO GR Date'].isna()]
    POs_no_ETA = POs_no_ETA[POs_no_ETA['ETA'].isna()]
    POs_no_ETA['current_date'] = pd.to_datetime(date.today())
    POs_no_ETA.drop(['ETA', 'Field PO GR Date'], axis=1, inplace=True)
    POs_no_ETA['Field PO Date'] = pd.to_datetime(POs_no_ETA['Field PO Date'])
    POs_no_ETA['elap_time_days'] = (POs_no_ETA.current_date - POs_no_ETA['Field PO Date']).dt.days

    ETA_POs = df.drop(['Field PO Request Date(RDD)','OBS Creation Date HUB', 'Field PO GR Date'], axis=1)
    ETA_POs = ETA_POs[(ETA_POs.ETA.notna())]
    ETA_POs = ETA_POs[ETA_POs['Field PO Date']> '2022-01-01']
    ETA_POs = ETA_POs[~(ETA_POs['ETA'] == datetime.strptime('00:00:00', '%H:%M:%S').time())]
    ETA_POs = ETA_POs[~(ETA_POs.ETA == '0-Jan-00')]
    ETA_POs.ETA = pd.to_datetime(ETA_POs.ETA)
    ETA_POs['time_ETA'] = ETA_POs.ETA - ETA_POs['Field PO Date']
    ETA_POs.time_ETA = ETA_POs.time_ETA.dt.days
    
    st.markdown('#### Open POs without ETA from creation to current date')
    vendor = st.sidebar.selectbox('VENDOR ID:',POs_no_ETA['Field PO Vendor'][POs_no_ETA['Field PO Vendor'].notnull()].unique())
    creation_date = st.text_input('POs Creation Date: YYYY-MM-DD')
    elap_days = st.slider('Elapsed Days Today',min(POs_no_ETA['elap_time_days'][POs_no_ETA['Field PO Date'] > creation_date]), max(POs_no_ETA['elap_time_days'][POs_no_ETA['Field PO Date'] > creation_date]), int(np.mean(POs_no_ETA['elap_time_days'][POs_no_ETA['Field PO Date'] > creation_date])))

    POs_no_ETA = POs_no_ETA[(POs_no_ETA['Field PO Date'] > creation_date) & (POs_no_ETA['Field PO Vendor'] == vendor) & (POs_no_ETA['elap_time_days'] > elap_days)]
    POs_no_ETA.drop(['current_date','HUB SO Process Class'], axis=1, inplace=True)
    
    st.write('Number of POs: ' + str(POs_no_ETA.elap_time_days.count()))
    st.dataframe(POs_no_ETA)

    st.markdown('#### Open POs with ETA')
    ETA_date_min = st.text_input('ETA_start: YYYY-MM-DD')
    ETA_date_max = st.text_input('ETA_end: YYYY-MM-DD')
    
    #time_ETA = st.slider('ETA', min(ETA_POs['time_ETA'][ETA_POs['ETA'] > ETA_date]), max(ETA_POs['time_ETA'][ETA_POs['ETA'] > ETA_date]), int(np.mean(ETA_POs['time_ETA'][ETA_POs['ETA'] > ETA_date])))
    ETA_POs =  ETA_POs[(ETA_POs['ETA'] > ETA_date_min) & (ETA_POs['Field PO Vendor'] == vendor) & (ETA_POs['ETA'] < ETA_date_max)]
    ETA_POs.drop(['Field PO Date','HUB SO Process Class'], axis=1, inplace=True)
    st.write('Number of POs: ' + str(ETA_POs.time_ETA.count()))

    st.dataframe(ETA_POs)

st.sidebar.title('Analytics')
options = st.sidebar.radio('SHEETS',options=['Home','GR POs','No ETA POs','ETA POs','Tracking'])

if options == 'GR POs':
    gr_pos(Gold_data)
elif options == 'No ETA POs':
    no_eta_pos(Gold_data)
elif options == 'Tracking':
    tracking_pos(Gold_data)
elif options == 'ETA POs':
    eta_pos(Gold_data)
elif options == 'Home':
    home()
    describe(Gold_data)
