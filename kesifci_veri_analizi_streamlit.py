#################### KEŞİFÇİ VERİ ANALİZİ STREAMLIT UYGULAMA ###################
# Uygulama - Kod : Mustafa Gürkan Çanakçı
# Kaynak : https://docs.streamlit.io/library/api-reference/performance

################################# IMPORT LIBRARIES ###############################################################

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import pathlib
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from pathlib import Path
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, RandomizedSearchCV, validation_curve
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve, classification_report, precision_recall_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split, cross_validate
from matplotlib.gridspec import GridSpec

st.set_page_config(page_title="EDA,Visualization and Modeling",
                   layout="wide")


########################## 1. DOSYA YÜKLEME ALANI  #######################################################33
uploaded_file = st.file_uploader("Choose a CSV file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = uploaded_file.getvalue().decode('utf-8').splitlines()
    st.session_state["preview"] = ''
    for i in range(0, min(5, len(data))):
        st.session_state["preview"] += data[i]

def upload():
    if uploaded_file is None:
        st.session_state["upload_state"] = "Upload a file first!"
    else:
        data = uploaded_file.getvalue().decode('utf-8')
        parent_path = pathlib.Path(__file__).parent.parent.resolve()
        save_path = os.path.join(parent_path, "data")
        complete_name = os.path.join(save_path, uploaded_file.name)
        destination_file = open(complete_name, "w")
        destination_file.write(data)
        destination_file.close()


# ############### 2. FONKSİYONLARI EKLİYORUZ #####################################
@st.cache
def get_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

def check_data(dataframe, head=5):

        st.subheader("**Veri Setinin Analizi**")

        st.markdown("**Veri Setinin İlk 5 Değeri**")
        dataframe1 = dataframe.head()
        st.dataframe(dataframe1)

        st.markdown("**Veri Setinin Son 5 Değeri**")
        dataframe2 = dataframe.head()
        st.dataframe(dataframe2)

        st.markdown("**Eksik Değerler**")
        dataframe3 = dataframe.isnull().sum()
        st.write(dataframe3)

        st.markdown("**Betimsel İstatistik**")
        dataframe4 = dataframe.describe([0.01, 0.05, 0.10, 0.50, 0.75, 0.90, 0.95, 0.99]).T
        st.write(dataframe4)

        st.write("**Veri Boyutu:**", dataframe.shape)


#################### 2.2.Kategorik ve Numerik Değişkenler Fonksiyon #########################################3

def grab_col_names(dataframe, cat_th=10, car_th=20):

        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if
                       dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if
                       dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]



        return cat_cols, num_cols, cat_but_car

        cat_cols, num_cols, cat_but_car = grab_col_names(df)
        pass

def cat_summary(dataframe, col_name):
    dataframe1 = pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)})

    fig = Figure()
    fig.set_size_inches(16, 6)
    ax = fig.subplots()
    sns.countplot(
        x=dataframe[col_name], data=dataframe,
        ax=ax,
    )
    ax.set_ylabel("Count")

    kategori.pyplot(fig)
    kategori.dataframe(dataframe1)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    dataframe1 = dataframe[numerical_col].describe(quantiles).T

    fig = Figure()
    fig.set_size_inches(16, 6)
    ax = fig.subplots()
    sns.histplot(
        x=dataframe[numerical_col],data=dataframe,
        bins=20,
        ax=ax,
    )

    sayisal.pyplot(fig)
    sayisal.dataframe(dataframe1)

#################### 2.3.GÖRSELLEŞTİRMEYE DAYALI KULLANDIĞIMIZ FONKSİYONLAR #########################################3

def countpl(dataframe,col):
    fig = Figure()
    fig.set_size_inches(16, 6)
    ax = fig.subplots()
    sns.countplot(
        y=dataframe[col], data=dataframe,
        ax=ax,
    )
    ax.set_xlabel("Count")
    st.pyplot(fig)

def scatterpl(dataframe,col1,col2):
    fig = Figure()
    fig.set_size_inches(16, 6)
    ax = fig.subplots()
    sns.scatterplot(
        x=dataframe[col1], y=dataframe[col1],hue=dataframe[col2],data= dataframe,
        ax=ax,
    )
    st.pyplot(fig)


def histpl(dataframe,col1,col2):
    fig = Figure()
    fig.set_size_inches(10,6)
    ax = fig.subplots()
    sns.histplot(data= dataframe,x=dataframe[col1], hue=dataframe[col2],ax=ax,)

    st.pyplot(fig)

def boxpl(dataframe,col1,col2):
    fig = Figure()
    fig.set_size_inches(16, 6)
    ax = fig.subplots()
    sns.boxplot(x=dataframe[col1], y=dataframe[col2],hue=dataframe[col2],data= dataframe,
        ax=ax,
    )
    st.pyplot(fig)



def violinplot(dataframe,target):
# Select columns for plotting
    columns = df.columns[:-1]

# Define color palette
    colors_list = ['#78C850', '#F08030']

# Set up figure and axes
    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(15, 17))
    axs = axs.ravel()

# Loop over columns and plot
    for i, col in enumerate(columns):
        sns.violinplot(
            x=target, y=col, data=df, kind='violin',
            split=True, height=4, aspect=.7, palette=colors_list,
            ax=axs[i]
        )
        sns.swarmplot(
            x=target, y=col, data=df, color='k', alpha=0.8,
            ax=axs[i]
        )
# Set titles and axis labels
    for i, ax in enumerate(axs):
        ax.set_title(columns[i])
        ax.set_xlabel('Species')
        ax.set_ylabel(columns[i])

# Remove empty plots
    fig.delaxes(axs[-1])
    fig.delaxes(axs[-2])

# Add spacing between plots
    fig.tight_layout()

# Show figure in Streamlit app
    st.pyplot(fig)


def histoplot(dataframe):
    columns = dataframe.columns.tolist()
    columns.pop()

    # define colors for the graphs
    colours = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'b']

    # set seaborn style and figure size
    sns.set(rc={'figure.figsize': (15, 17)})
    sns.set_style(style='white')

    # create figure and axes
    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 17))
    axes = axes.flatten()

    # plot each column as a histogram with corresponding color
    for i, column in enumerate(columns):
        sns.histplot(dataframe[column], ax=axes[i], kde=True, color=colours[i])
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Density')
        axes[i].grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))

    # remove any extra axes
    for i in range(len(columns), len(axes)):
        fig.delaxes(axes[i])

    # set title and layout for the plot
    fig.suptitle('Distribution of Features', fontsize=20, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    st.pyplot(fig)


def special_visual(dataframe,target):
    fig = plt.figure(figsize=(24, 10), dpi=60)
    gs = GridSpec(ncols=10, nrows=12, left=0.05, right=0.5, wspace=0.2, hspace=0.1)
    fig.patch.set_facecolor('#f5f5f5')
    sns.set_palette(sns.color_palette(['#00f5d4', '#f15bb5']))

    ax1 = fig.add_subplot(gs[1:6, 0:4])
    ax3 = fig.add_subplot(gs[1:6, 5:])
    ax4 = fig.add_subplot(gs[8:, 0:])

    axes = [ax1, ax3, ax4]

    for ax in axes:
        ax.axes.get_yaxis().set_visible(False)
        ax.set_facecolor('#f5f5f5')
        for loc in ['left', 'right', 'top', 'bottom']:
            ax.spines[loc].set_visible(False)

    sns.kdeplot(x='Glucose', data=dataframe[dataframe[target] == 0], ax=ax1, shade=True, color='#00f5d4', alpha=1)
    sns.kdeplot(x='Glucose', data=dataframe[dataframe[target] == 1], ax=ax1, shade=True, color='#b30f72', alpha=0.8)
    ax1.set_xlabel('Average Glucose Level', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': 'black'})
    ax1.text(-20, 0.0175, 'Glucose Distribution by Outcome',
             {'font': 'cursive', 'size': '20', 'color': 'black', 'weight': 'bold'})
    ax1.text(150, 0.014, 'Diabet', {'font': 'cursive', 'fontsize': 16, 'color': '#b30f72'})
    ax1.text(200, 0.014, '|', {'font': 'Serif', 'fontsize': 16, 'fontweight': 'bold', 'color': 'black'})
    ax1.text(225, 0.014, 'Healthy', {'font': 'cursive', 'fontsize': 16, 'color': '#00c5a4'})

    sns.kdeplot(x='BMI', data=dataframe[dataframe[target] == 0], ax=ax3, shade=True, color='#00f5d4', alpha=1)
    sns.kdeplot(x='BMI', data=dataframe[dataframe[target] == 1], ax=ax3, shade=True, color='#b30f72', alpha=0.8)
    ax3.set_xlabel('bmi', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': 'black'})
    ax3.text(10, 0.070, 'Bmi Distribution by Outcome',
             {'font': 'cursive', 'size': '20', 'color': 'black', 'weight': 'bold'})
    ax3.text(45, 0.062, 'Diabet', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': '#b30f72'})
    ax3.text(60, 0.062, '|', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': 'black'})
    ax3.text(63, 0.062, 'Healthy', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': '#00c5a4'})

    palette = {0: '#00f5d4', 1: '#b30f72'}

    sns.scatterplot(data=dataframe, x=dataframe['Glucose'], y=dataframe['BMI'], hue=target, ax=ax4,
                    palette=palette)
    ax4.set_xlabel('Glucose Level', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': 'black'})
    ax4.set_ylabel('Glucose Level', {'font': 'cursive', 'fontsize': 16, 'fontweight': 'bold', 'color': 'black'})
    ax4.text(50, 0.0175, 'Glucose Distribution by Outcome',
             {'font': 'cursive', 'size': '20', 'color': 'black', 'weight': 'bold'})

    st.pyplot(fig)

########################### FEATURE ############################

def outlier_th(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    a, b = st.columns(2)

    a.text(col_name + " " + "Üst Limit")
    a.write(up_limit)
    b.text(col_name+ " " + "Alt Limit")
    b.write(low_limit)


    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_th(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_th(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit



def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    st.write(missing_df, end="\n")

    if na_name:
        return na_columns

####################### ENCODING - SCALING #####################################################

def encode_data(dataframe):
    le = LabelEncoder()
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float]
                   and dataframe[col].nunique() == 2]
    for col in binary_cols:
        dataframe[col] = le.fit_transform(dataframe[col])
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = pd.get_dummies(dataframe, columns=ohe_cols, drop_first=True)

    return dataframe
    pass

def scale(dataframe):
    scaler = RobustScaler()
    dataframe[num_cols] = scaler.fit_transform(df[num_cols])
    st.write(dataframe)
    return dataframe


                

###########################  3. BAŞLIKLARIMIZ ############################


# Define the sidebar sections
sections = ['Exploratory data analysis', 'Data Visualization', 'Data Preprocessing','Create Modeling']

# Define the sidebar options for each section
section_options = {
    'Exploratory data analysis': ['Check data'],
    'Data Visualization': ['Visualization','Special Visual'],
    'Data Preprocessing': ['Preprocessing'],
    'Create Modeling': ['Model']
}

# Set the default section and option
default_section = 'Exploratory data analysis'
default_option = 'Check data'

# Define the sidebar
sidebar = st.sidebar
selected_section = sidebar.selectbox('Select a section', sections, index=0)
selected_option = sidebar.selectbox('Select an option', section_options[selected_section], index=0)



data = get_data(uploaded_file)
df = data.copy()
cat_cols, num_cols, cat_but_car = grab_col_names(df)

if selected_option == 'Check data':
        st.title("Exploratory Data Analysis")
        st.image("https://datos.gob.es/sites/default/files/u322/grafico.jpg")
        st.header('Veri Setimiz')
        st.markdown('Veri setini yükledikten sonra inceliyoruz.')
        df1 = df.head()
        st.dataframe(df)

        st.header('Keşifçi Veri Analizi')
        st.markdown('**Bu bölümde veri setimizi analiz edeceğiz.**')

        # Veri Setinin Check Ediyoruz.
        check = check_data(df)

        cat_cols, num_cols, cat_but_car = grab_col_names(df)

        st.subheader("**Değişken Analizi**")

        st.write("Toplam Gözlem Sayısı :", df.shape[0])
        st.write("Toplam Değişken Sayısı : ", df.shape[1])

        degisken1, degisken2, degisken3 = st.columns(3)

        degisken1.markdown("**Kategorik Değişkenler**")
        degisken1.info(cat_cols)
        degisken1.text("Değişken Sayısı")
        degisken1.write(len(cat_cols))

        degisken2.markdown("**Numerik Değişkenler**")
        degisken2.info(num_cols)
        degisken2.text("Değişken Sayısı")
        degisken2.write(len(num_cols))

        degisken3.markdown("**Kardinal Değişkenler**")
        degisken3.info(cat_but_car)
        degisken3.text("Değişken Sayısı")
        degisken3.write(len(cat_but_car))

        kategori, sayisal = st.columns(2)

        kategori.subheader("**Kategorik Değişkenlerin Görselleştirmesi**")
        kategori.markdown("Kategorik değişkenleri hem görsel hem de istatistiki açıdan inceleyelim. ")

        for col in cat_cols:
            cat_summary(df, col)

        sayisal.subheader("**Numerik Değişkenlerin Görselleştirmesi**")
        sayisal.markdown("Sayısal değişkenlerimizi hem görsel hem de istatistiki açıdan gözlemleyelim.")

        for col in num_cols:
            num_summary(df, col)


elif selected_option == 'Visualization':


    top1,top2= st.columns([2,1])

    with top1:
        st.header('Veri Görselleştirme')
        st.markdown("**Veri görselleştirmeye dair grafiksel pivot tablolarımızı gösteriyoruz.**")
        chart_choice = st.selectbox("", ["Scatter Plot", "Count Plot", "Histplot", "Box Plot"])


        if chart_choice == "Count Plot":
            for col in cat_cols:
                countpl(df, col)

        if chart_choice == "Histplot":
            hue = st.selectbox('Kategorik Değişkenler', options=cat_cols)
            for col in num_cols:
                histpl(df, col1=col, col2=hue)

        if chart_choice == "Scatter Plot":
            hue = st.selectbox('Kategorik Değişkenler', options=cat_cols)
            for col in num_cols:
                scatterpl(df, col, col2=hue)

        if chart_choice == "Box Plot":
            hue = st.selectbox('Kategorik Değişkenler', options=cat_cols)
            for col in num_cols:
                boxpl(df, col1=col, col2=hue)

    with top2:
        st.header('Korelasyon')
        st.markdown("**Sayısal değişkenlerin arasındaki ilişkiyi görmekteyiz.**")

        corr = df[num_cols].corr()

        sns.heatmap(corr, annot=True)

        fig = Figure()
        fig.set_size_inches(20, 10)
        ax = fig.subplots()
        sns.heatmap(corr, cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
                    annot=True, annot_kws={"size": 10}, square=True, ax=ax, )

        st.pyplot(fig)

        st.markdown("**Değişkenlerin Analiz Gösterimi**")

        plt.style.use('ggplot')

        # Set up the plot using plt.subplots()
        fig, ax = plt.subplots(figsize=(11, 15))

        # Customize the plot
        ax.set_facecolor('#fafafa')
        ax.set(xlim=(-.05, 200))
        ax.set_ylabel('Değişkenler')
        ax.set_title("Veri Setine Genel Bakış")

        # Create the plot using Seaborn's boxplot() function
        sns.boxplot(data=df,
                    orient='h',
                    palette='Set2',
                    ax=ax)

        # Display the plot using Streamlit's st.pyplot() function
        st.pyplot(fig)

elif selected_option == "Special Visual":

    st.title("Special Visualization")
    st.header("Boxen Plot Various Features")
    st.markdown("Bu alanda özel görselleştirme grafiklerimizi göstereceğiz.")
    # Create the plot using fig, ax
    fig = plt.figure(figsize=(18, 15))
    gs = fig.add_gridspec(3, 3)
    gs.update(wspace=0.5, hspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])


    background_color = "#0e7b92"
    # c9c9ee
    color_palette = ["#f56476", "#ff8811", "#ff0040", "#ff7f6c", "#f0f66e", "#990000"]
    fig.patch.set_facecolor(background_color)
    ax0.set_facecolor(background_color)
    ax1.set_facecolor(background_color)
    ax2.set_facecolor(background_color)
    ax3.set_facecolor(background_color)
    ax4.set_facecolor(background_color)
    ax5.set_facecolor(background_color)
    ax6.set_facecolor(background_color)
    ax7.set_facecolor(background_color)
    ax8.set_facecolor(background_color)

    # Title of the plot
    ax0.spines["bottom"].set_visible(False)
    ax0.spines["left"].set_visible(False)
    ax0.spines["top"].set_visible(False)
    ax0.spines["right"].set_visible(False)
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])

    ax0.text(0.5, 0.5,
             'Boxenplot plot for various\n features\n_________________\n\n CREDIT: Gurkan',
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=18, fontweight='bold',
             fontfamily='serif',
             color="#FFFFFF")

    # Pregnancies
    ax1.text(-0.18, 19, 'Pregnancies', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax1.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax1, y=df['Pregnancies'], palette=["#f56476"], width=0.6)
    ax1.set_xlabel("")
    ax1.set_ylabel("")

    # Glucose
    ax2.text(-0.1, 217, 'Glucose', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax2.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax2, y=df['Glucose'], palette=["#ff8811"], width=0.6)
    ax2.set_xlabel("")
    ax2.set_ylabel("")

    # BloodPressure
    ax3.text(-0.20, 140, 'BloodPressure', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax3.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax3, y=df['BloodPressure'], palette=["#ff0040"], width=0.6)
    ax3.set_xlabel("")
    ax3.set_ylabel("")

    # SkinThickness
    ax4.text(-.2, 112, 'SkinThickness', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax4.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax4, y=df['SkinThickness'], palette=["#ff7f6c"], width=0.6)
    ax4.set_xlabel("")
    ax4.set_ylabel("")

    # Insulin
    ax5.text(-0.10, 1000, 'Insulin', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax5.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax5, y=df['Insulin'], palette=["#f0f66e"], width=0.6)
    ax5.set_xlabel("")
    ax5.set_ylabel("")

    # BMI
    ax6.text(-0.08, 75, 'BMI', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax6.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax6, y=df['BMI'], palette=["#990000"], width=0.6)
    ax6.set_xlabel("")
    ax6.set_ylabel("")

    # DPF
    ax7.text(-0.065, 2.8, 'DPF', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax7.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax7, y=df['DiabetesPedigreeFunction'], palette=["#3339FF"], width=0.6)
    ax7.set_xlabel("")
    ax7.set_ylabel("")

    # Age
    ax8.text(-0.08, 86, 'Age', fontsize=14, fontweight='bold', fontfamily='serif', color="#FFFFFF")
    ax8.grid(color='#FFFFFF', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
    sns.boxenplot(ax=ax8, y=df['Age'], palette=["#34495E"], width=0.6)
    ax8.set_xlabel("")
    ax8.set_ylabel("")

    # Remove spines from ax1 and ax2
    for s in ["top", "right", "left", "bottom"]:
        ax1.spines[s].set_visible(False)
        ax2.spines[s].set_visible(False)
        ax3.spines[s].set_visible(False)
        ax4.spines[s].set_visible(False)
        ax5.spines[s].set_visible(False)
        ax6.spines[s].set_visible(False)
        ax7.spines[s].set_visible(False)
        ax8.spines[s].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])


    # Convert the plot to a Streamlit component
    st.pyplot(fig)


    histoplot(df)

    st.header("Outcome by Various Factors")
    violinplot(df,"Outcome")

    special_visual(df,"Outcome")


elif selected_option == "Preprocessing":
    st.header("Encoding Data")
    processed_df = encode_data(df)
    st.write("Processed data:")
    st.write(processed_df)

    st.header("Scaling Data")
    st.write("Scaling data:")
    scale(processed_df)


elif selected_option == 'Model':
    st.header("Modeling")
    st.text("Modelleme öncesinde scale etmiş olduğumuz ilgili veri setini oluşturuyoruz.")
    processed_df = encode_data(df)
    scale(processed_df)


    X = processed_df.drop('Outcome', axis=1)
    y = processed_df['Outcome']

    # Sidebar for Model Selection
    st.sidebar.subheader("Model Selection")
    models = {"Logistic Regression": LogisticRegression,
              "Decision Tree": DecisionTreeClassifier,
              "Random Forest": RandomForestClassifier,
              "AdaBoost": AdaBoostClassifier,
              "K-Nearest Neighbors": KNeighborsClassifier,
              "Gradient Boosting": GradientBoostingClassifier,
              "XGBoost": XGBClassifier,
              "LightGBM": LGBMClassifier,
              "CatBoost": CatBoostClassifier}

    model_name = st.sidebar.selectbox("Select a model to run", list(models.keys()))
    st.text("Modelimizi train/test seti olarak ayırıyoruz. ")
    # Model Building / Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
    model = models[model_name]()
    st.title(model)

    model.fit(X_train, y_train)

    # Sidebar for Feature Importance
    st.sidebar.subheader("Feature Importance")
    show_importance = st.sidebar.checkbox("Show Feature Importance")

    # Prediction and Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Accuracy: {acc}")

    class_report = classification_report(y_test, y_pred)
    st.write(f"Classification report:\n \n \n {class_report}  ")

    auc = roc_auc_score(y_test, y_pred)
    st.write(f"AUC score: {auc}")


    predicted_probab_log = model.predict_proba(X_test)
    predicted_probab_log = predicted_probab_log[:, 1]
    fpr, tpr, _ = roc_curve(y_test, predicted_probab_log)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker='.', label=model)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend()
    st.pyplot(fig)

    # Cross validation

    st.header("5-Fold Cross Validation")
    cv_results = cross_validate(model, X, y, cv=5, scoring=["accuracy", "precision", "recall", "f1", "roc_auc"])

    st.write(f"Test accuracy (CV): {cv_results['test_accuracy'].mean()}")
    st.write(f"Test f1 score (CV): {cv_results['test_f1'].mean()}")
    st.write(f"Test ROC AUC score (CV): {cv_results['test_roc_auc'].mean()}")


