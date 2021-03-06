import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import msoffcrypto
from PIL import Image

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["password"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct.
        return True

if check_password():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.title('Asia Market Clustering Analysis for Product Protection')
    st.write('created by Mason Choi (last updated 2022-06-07)')
    passwd = st.secrets['passwd']


    @st.cache
    def read_excel():
        decrypted_workbook = io.BytesIO()
        with open('./data/df_data_kmean.xlsx', 'rb') as file:
            office_file = msoffcrypto.OfficeFile(file)
            office_file.load_key(password=passwd)
            office_file.decrypt(decrypted_workbook)
            df = pd.read_excel(decrypted_workbook, sheet_name='Sheet1', index_col=0, engine='openpyxl')
            return df


    @st.cache
    def read_data_a():
        preservative = pd.read_csv('./data/df_preservative.csv', index_col=0)
        antioxidant = pd.read_csv('./data/df_antioxidant.csv', index_col=0)
        chelating = pd.read_csv('./data/df_chelating.csv', index_col=0)
        evonik = pd.read_csv('./data/df_evonik.csv', index_col=0)
        tsne = pd.read_csv('./data/df_tsne.csv', index_col=0)

        return preservative, antioxidant, chelating, evonik, tsne

    df_data = read_excel()
    df_preservative, df_antioxidant, df_chelating, df_evonik, df_tsne = read_data_a()

    all = ['all']
    select_option = ['nothing', 'preservative', 'p_system', 'antioxidant', 'a_system', 'chelating', 'c_system',
                     'evonik', 'e_sytem']

    with st.form(key='my_form1'):
        st.subheader('Market selection')
        col1_1, col1_2 = st.columns([3, 1])
        with col1_1:
            market = st.selectbox('Market', all + list(df_data['Market'].unique()))
            manufacturer = st.selectbox('Manufacturer', all + list(df_data['Manufacturer'].unique()))
        with col1_2:
            market_max = st.number_input('market show', min_value=1, max_value=20, value=15)
            manufacturer_max = st.number_input('Manufacturer show', min_value=1, max_value=30, value=15)
        year = st.multiselect('year', all + list(df_data['Year'].unique()))

        col2_1, col2_2, col2_3 = st.columns(3)
        with col2_1:
            option_cat = st.radio('show by', ['Category', 'Class', 'Sub-Category'])
        with col2_2:
            category = st.selectbox('category', all + list(df_data['Category'].unique()))
        with col2_3:
            class_ = st.selectbox('class', all + list(df_data['Class'].unique()))

        col2_1, col2_2 = st.columns([1,2])
        with col2_2:
            sub_category = st.selectbox('sub-category', all + list(df_data['Sub-Category'].unique()))

        submit_button = st.form_submit_button(label='Submit')

    @st.cache
    def mask_generator(string, column):
        if string != 'all':
            return df_data[column] == string
        else:
            return np.array([True] * len(df_data))


    mask_market = mask_generator(market, 'Market')
    mask_manufacturer = mask_generator(manufacturer, 'Manufacturer')

    if option_cat == 'Category':
        mask_cat = mask_generator(category, 'Category')
    elif option_cat == 'Class':
        mask_cat = mask_generator(class_, 'Class')
    else:
        mask_cat = mask_generator(sub_category, 'Sub-Category')

    if 'all' in year or len(year) == 0:
        mask_year = np.array([True] * len(df_data))
    else:
        mask_year = np.array([False] * len(df_data))
        for y in year:
            mask_year = (mask_year) | (df_data['Year'] == y)

    mask_final = (mask_market) & (mask_manufacturer) & (mask_cat) & (mask_year)

    df_data = df_data[mask_final]
    df_preservative = df_preservative[mask_final]
    df_antioxidant = df_antioxidant[mask_final]
    df_chelating = df_chelating[mask_final]
    df_evonik = df_evonik[mask_final]
    df_tsne = df_tsne[mask_final]

    with st.form(key='my_form2'):
        st.subheader('ingredient selection')
        col3_1, col3_2, col3_3, = st.columns([1, 2, 2])
        with col3_1:
            option_product = st.radio('select options', select_option)

        with col3_2:
            one_p = st.selectbox('preservative', df_preservative.columns)
            one_a = st.selectbox('antioxidant', df_antioxidant.columns)
            one_c = st.selectbox('chelating', df_chelating.columns)
            one_e = st.selectbox('evonik', df_evonik.columns)

        with col3_3:
            one_p_system = st.selectbox('p_system', df_data['p_system'].value_counts().index)
            one_a_system = st.selectbox('a_system', df_data['a_system'].value_counts().index)
            one_c_system = st.selectbox('c_system', df_data['c_system'].value_counts().index)
            one_e_system = st.selectbox('e_system', df_data['e_system'].value_counts().index)

        submit_button = st.form_submit_button(label='Submit')


    def mask_generator2(string, df, column=None):
        if column:
            mask = df[column] == string
        else:
            mask = df[string] == 1
        return mask


    one = [one_p, one_p_system, one_a, one_a_system, one_c, one_c_system, one_e, one_e_system]
    selection_ingredient = 'nothing'
    if option_product == select_option[1]:
        mask_pf = mask_generator2(one_p, df_preservative)
        selection_ingredient = one_p
    elif option_product == select_option[2]:
        mask_pf = mask_generator2(one_p_system, df_data, column='p_system')
        selection_ingredient = one_p_system
    elif option_product == select_option[3]:
        mask_pf = mask_generator2(one_a, df_antioxidant)
        selection_ingredient = one_a
    elif option_product == select_option[4]:
        mask_pf = mask_generator2(one_a_system, df_data, column='a_system')
        selection_ingredient = one_a_system
    elif option_product == select_option[5]:
        mask_pf = mask_generator2(one_c, df_chelating)
        selection_ingredient = one_c
    elif option_product == select_option[6]:
        mask_pf = mask_generator2(one_c_system, df_data, column='c_system')
        selection_ingredient = one_c_system
    elif option_product == select_option[7]:
        mask_pf = mask_generator2(one_e, df_evonik)
        selection_ingredient = one_e
    elif option_product == select_option[8]:
        mask_pf = mask_generator2(one_e_system, df_data, column='e_system')
        selection_ingredient = one_e_system
    else:
        mask_pf = [True] * len(df_data)

    df_data = df_data[mask_pf]
    df_tsne = df_tsne[mask_pf]
    df_preservative = df_preservative[mask_pf]
    df_antioxidant = df_antioxidant[mask_pf]
    df_chelating = df_chelating[mask_pf]
    df_evonik = df_evonik[mask_pf]


    # =================================================================================
    # plot functions

    def plot_1(data, subject, size=(10, 4), max=None):
        st.subheader(subject)
        fig, ax = plt.subplots(figsize=size)
        if max:
            sr_subject = data[subject].value_counts()[:max]
        else:
            sr_subject = data[subject].value_counts()

        bars = ax.bar(sr_subject.index, sr_subject, color=(153 / 255, 29 / 255, 133 / 255))
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() * (1 / 2), b.get_height() + 1, sr_subject[i], ha='center', fontsize=13)
            ax.text(b.get_x() + b.get_width() * (1 / 2), b.get_height() * (1 / 2),
                    round(sr_subject[i] / len(data) * 100, 2),
                    ha='center', fontsize=10)
        plt.xticks(rotation=90)
        return fig


    def plot_2(data, size=(10, 4), max=10):
        st.subheader('Most frequent')
        data = data.drop('Year', axis=1)
        sr_sum = data.sum(axis=0).sort_values(ascending=False)[:max]

        fig, ax = plt.subplots(figsize=size)
        bars = ax.bar(sr_sum.index, sr_sum, color=(153 / 255, 29 / 255, 133 / 255))
        for i, b in enumerate(bars):
            ax.text(b.get_x() + b.get_width() * (1 / 2), b.get_height() + 1, sr_sum[i], ha='center', fontsize=13)
            ax.text(b.get_x() + b.get_width() * (1 / 2), b.get_height() * (1 / 2),
                    round(sr_sum[i] / len(data) * 100, 2),
                    ha='center', fontsize=10)
        plt.xticks(rotation=90)
        return fig


    def plot_3(data2, max=10):
        st.subheader('Trend')
        columns = data2.sum(axis=0).sort_values(ascending=False).index[1:max]
        df_group = data2.groupby('Year')[columns].sum().T
        c1, c2 = st.columns([1,3])
        with c1:
            st.table(data2['Year'].value_counts().sort_index())
        with c2:
            st.table(df_group)
        df_group = df_group / data2.groupby('Year').size() * 100
        df_group.T.plot(figsize=(7, 5))
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        st.pyplot()
    # ===============================================================================

    # clustering map 1 and 2
    if len(df_data) == 0:
        df_cluster = 'no product'
        kmeans_list = []
    else:
        df_cluster = df_data.groupby(['kmeans','x','y','Sub-Category'])['w'].aggregate(
                    ['count', 'mean']).sort_values('count', ascending=False)
        df_cluster.reset_index(level=['x', 'y', 'Sub-Category'], inplace=True)
        kmeans_list = list(df_data['kmeans'].value_counts().index)


    with st.form(key='my_form3'):
        st.subheader('cluster selection')
        st.write(df_cluster)
        selection_kmean = st.multiselect('cluster', all + kmeans_list)
        submit_button = st.form_submit_button(label='Submit')

    if 'all' in selection_kmean or len(selection_kmean) == 0:
        df_tsne_old = df_tsne.copy()
    else:
        mask_kmean = [False] * len(df_data)
        for k in selection_kmean:
            mask_kmean = (mask_kmean) | (df_data['kmeans'] == k)

        # mask_kmean = df_data['kmeans'] == selection_kmean

        df_data = df_data[mask_kmean]
        df_preservative = df_preservative[mask_kmean]
        df_antioxidant = df_antioxidant[mask_kmean]
        df_chelating = df_chelating[mask_kmean]
        df_tsne_old = df_tsne.copy()
        df_tsne = df_tsne[mask_kmean]

    col_fig_1, col_fig_2 = st.columns(2)
    with col_fig_1:
        img = Image.open("./data/clustering_gray.png")
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(img, extent=[-61.5, 61.5, -61.5, 61.5])
        if len(df_data) == 97958 or len(df_data) == 0:
            pass
        else:
            ax.scatter(df_tsne_old.iloc[:, 0], df_tsne_old.iloc[:, 1], c='red', marker='v', s=25, alpha=1)
            ax.scatter(df_tsne.iloc[:, 0], df_tsne.iloc[:, 1], c='orange', marker='v',
                       edgecolor='black', linewidth=2, s=150, alpha=1)

        plt.grid(True)
        plt.xticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60])
        plt.yticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60])
        st.pyplot(fig)

    with col_fig_2:
        img = Image.open("./data/clustering.png")
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(img, extent=[-61.5, 61.5, -61.5, 61.5])
        if len(df_data) == 97958 or len(df_data) == 0:
            pass
        else:
            ax.scatter(df_tsne_old.iloc[:, 0], df_tsne_old.iloc[:, 1], c='red', marker='v', s=25, alpha=1)
            ax.scatter(df_tsne.iloc[:, 0], df_tsne.iloc[:, 1], c='orange', marker='v',
                       edgecolor='black', linewidth=2, s=150, alpha=1)
        plt.grid(True)
        plt.xticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60])
        plt.yticks([-60, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60])
        st.pyplot(fig)

    # selection summary
    st.subheader('data selection summary')
    a, b, c, d = st.columns(4)
    with a:
        st.write('Market :', market)
    with b:
        st.write('Manufacturer :', manufacturer)
    with c:
        st.write('no. of products: ', len(df_data))
    with d:
        st.write('selected cluster : ', selection_kmean)

    e, f = st.columns(2)
    with e:
        if option_cat == 'Category':
            st.write('Option_show : ', option_cat, '>', category)
        elif option_cat == 'Class':
            st.write('Option_show : ', option_cat, '>', class_)
        else:
            st.write('Option_show : ', option_cat, '>', sub_category)
    with f:
        st.write('ingredient selection : ', option_product, ' > ', selection_ingredient)

    if len(df_data) == 0:
        st.header('There is no product 😕, please try it agian')


    st.subheader('Overall')
    with st.expander('see more'):
        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.pyplot(plot_1(df_data, 'Market', size=(5, 4), max=int(market_max)))
        with col_o2:
            st.pyplot(plot_1(df_data, 'Category', size=(5, 4)))

        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.pyplot(plot_1(df_data, 'Sub-Category', size=(5, 4)))
        with col_o2:
            st.pyplot(plot_1(df_data, 'Brand', size=(5, 4), max=int(manufacturer_max)))

        col_o1, col_o2 = st.columns(2)
        with col_o1:
            st.pyplot(plot_1(df_data, 'Ultimate Company', size=(5, 4), max=int(manufacturer_max)))
        with col_o2:
            st.pyplot(plot_1(df_data, 'Manufacturer', size=(5, 4), max=int(manufacturer_max)))

    st.subheader('Preservative')
    with st.expander('see more'):
        st.pyplot(plot_2(df_preservative, max=10))
        plot_3(df_preservative)
        st.pyplot(plot_1(df_data, 'p_system', max=10))

    st.subheader('Antioxidant')
    with st.expander('see more'):
        st.pyplot(plot_2(df_antioxidant, max=10))
        plot_3(df_antioxidant)
        st.pyplot(plot_1(df_data, 'a_system', max=10))

    st.subheader('Chelating_Agent')
    with st.expander('see more'):
        st.pyplot(plot_2(df_chelating, max=10))
        plot_3(df_chelating)
        st.pyplot(plot_1(df_data, 'c_system', max=10))

    st.subheader('Customer Report')
    sub_category = df_data['Class'].unique()
    with st.expander('see more'):
        if manufacturer == 'all':
            st.write('please select a manufacturer')
        else:
            st.subheader(manufacturer)
            for sub in sub_category:
                st.text(sub)
                st.text(df_data[df_data['Class'] == sub]['p_system'].value_counts()[:5])
                st.write()

    st.subheader('Product Report')
    with st.expander('see more'):
        if len(df_data) > 150:
            st.write('please reduce the number of product below 150')
        else:
            for i in range(len(df_data)):
                st.write(i)
                st.write(df_data.iloc[i, 2], ' ', df_data.iloc[i, 1], ' [', df_data.iloc[i, 4], ' / ',
                         df_data.iloc[i, 5], '] ', df_data.iloc[i, 12], df_data.iloc[i]['Market'])
                st.write(df_data.iloc[i, 7], ' > ', df_data.iloc[i, 11], ' > ', df_data.iloc[i, 8])
                st.write('Mintel Link :', df_data.iloc[i, 9])
                st.write('Ingredients :', df_data.iloc[i, 10])
                st.write('p-system: ', df_data.iloc[i, 13])
                st.write('a-system: ', df_data.iloc[i, 14])
                st.write('e-system: ', df_data.iloc[i, 15])
                st.write('c-system: ', df_data.iloc[i, 16])
                st.write("")
