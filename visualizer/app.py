import random

import cv2
import pandas as pd
import plotly.express as px
import streamlit as st
import yaml


@st.cache
def read_data(input_dir):
    df = pd.read_csv(input_dir + "transactions_train.csv", dtype={'article_id': str, 'sales_channel_id': str})
    df['t_dat'] = pd.to_datetime(df['t_dat'])
    # 週単位にする。testを0としたいので+1
    df['n_weeks_ago'] = ((df['t_dat'].max() - df['t_dat']).dt.days // 7) + 1
    articles = pd.read_csv(input_dir + "articles.csv", dtype={'article_id': str})
    customers = pd.read_csv(input_dir + "customers.csv")
    return df, articles, customers


@st.cache
def get_sub_data(df, customers, articles, min_purchased_count):
    unique_customers = customers['customer_id'].unique()
    active_unique_customers = df.loc[df.groupby(['customer_id'])['article_id'].transform("count") > min_purchased_count, 'customer_id'].unique()
    unique_articles = articles['article_id'].unique()
    return unique_customers, active_unique_customers, unique_articles


def visualize_article(articles, unique_articles, image_dir):
    target_article_id = select_target_article(unique_articles)
    show_article_info(articles, target_article_id)
    show_article_image(target_article_id, image_dir)


def select_target_article(unique_articles):
    target_article_id = st.selectbox("Target Article ID", unique_articles)
    if (target_article_id != "") & (not target_article_id in unique_articles):
        st.error(f"{target_article_id} is not in the dataset. Please check the id is correct.")
    if st.button("Random Choice"):
        target_article_id = random.choice(unique_articles)
    return target_article_id


def show_article_info(articles, target_article_id):
    target_article_info = articles[articles['article_id'] == target_article_id].T.astype(str)
    st.markdown("### Article Information")
    st.dataframe(target_article_info)


def show_article_image(target_article_id, image_dir):
    filename = str(image_dir + f'{target_article_id[:3]}/{target_article_id}.jpg')
    img = cv2.imread(filename)[:, :, ::-1]
    st.image(img, use_column_width=True)


def visualize_customer(df, customers, unique_customers, active_unique_customers, num_sample, max_display_per_col, image_dir):
    target_customer_id = select_target_customer(unique_customers, active_unique_customers)
    show_customer_info(customers, target_customer_id)
    show_customer_transactions(df, target_customer_id)
    show_frequently_purchased_articles(df, target_customer_id, num_sample, max_display_per_col, image_dir)
    show_recently_purchased_articles(df, target_customer_id, num_sample, max_display_per_col, image_dir)


def visualize_prediction(pred_df, customers, unique_customers, max_display_per_col, image_dir):
    target_customer_id = select_target_customer(unique_customers, pred_df['customer_id'].unique())
    show_customer_info(customers, target_customer_id)
    show_pred_gt_articles(target_customer_id, pred_df, max_display_per_col, image_dir)


def select_target_customer(unique_customers, active_unique_customers):
    target_customer_id = st.text_input(
        "Target Customer ID",
        # value='e805d4c5a1f5b03312e4b98f29b8a61519ecac5eb01435013ad96413856c02dd',
        placeholder='Paste the target customer id'
    )
    # if not target_customer_id in unique_customers:
    #     st.error(f"{target_customer_id} is not in the dataset. Please check the id is correct.")
    if st.button("Random Choice"):
        target_customer_id = random.choice(active_unique_customers)
    return target_customer_id


def show_customer_info(customers, target_customer_id):
    target_customer_info = customers[customers['customer_id'] == target_customer_id].T.astype(str)
    st.markdown("### Customer Information")
    st.dataframe(target_customer_info)


def show_customer_transactions(df, target_customer_id):
    st.markdown("### Customer Transactions")
    _df = df.loc[df['customer_id'] == target_customer_id]
    st.dataframe(_df)
    fig = px.bar(
        _df.groupby(['n_weeks_ago', 'sales_channel_id'])['article_id'].count().reset_index().rename(columns={'article_id': 'purchase count'}),
        x='n_weeks_ago', y='purchase count', color='sales_channel_id',
        range_x=[df['n_weeks_ago'].max(), df['n_weeks_ago'].min()],
        color_discrete_map={'1': 'blue', '2': 'red'}
    )
    fig.update_traces(width=1)
    st.plotly_chart(fig)


def show_frequently_purchased_articles(df, target_customer_id, num_sample, max_display_per_col, image_dir):
    st.markdown("### Frequently Purchased Articles")
    purchased_sample = df.loc[df['customer_id'] == target_customer_id, 'article_id'].value_counts().head(num_sample)
    purchased_articles = purchased_sample.index
    purchased_count = purchased_sample.values

    col = st.columns(max_display_per_col)
    for i, article_id in enumerate(purchased_articles):
        j = i % max_display_per_col
        with col[j]:
            st.write(f"id: {article_id}")
            filename = str(image_dir + f'{article_id[:3]}/{article_id}.jpg')
            img = cv2.imread(filename)[:, :, ::-1]
            st.image(img, use_column_width=True)
            st.write(f"count: {purchased_count[i]}")


def show_recently_purchased_articles(df, target_customer_id, num_sample, max_display_per_col, image_dir):
    st.markdown("### Recently Purchased Articles")
    recently_purchased_sample = df.loc[df['customer_id'] == target_customer_id, ['t_dat', 'article_id']
                                       ].drop_duplicates().sort_values('t_dat', ascending=False).head(num_sample)
    recently_purchased_articles = recently_purchased_sample['article_id'].to_numpy()
    recently_purchased_date = recently_purchased_sample['t_dat'].dt.strftime("%Y-%m-%d").to_numpy()
    col = st.columns(max_display_per_col)
    for i, article_id in enumerate(recently_purchased_articles):
        j = i % max_display_per_col
        with col[j]:
            st.write(f"id:{article_id}")
            filename = str(image_dir + f'{article_id[:3]}/{article_id}.jpg')
            img = cv2.imread(filename)[:, :, ::-1]
            st.image(img, use_column_width=True)
            st.write(f"date: {recently_purchased_date[i]}")


def show_pred_gt_articles(target_customer_id, pred_df, max_display_per_col, image_dir):
    targets = pred_df.loc[pred_df['customer_id'] == target_customer_id].iloc[0]
    st.markdown("### Prediction")
    col = st.columns(max_display_per_col)
    for i, article_id in enumerate(targets['item']):
        j = i % max_display_per_col
        with col[j]:
            st.write(f"id:{article_id}")
            filename = str(image_dir + f'{article_id[:3]}/{article_id}.jpg')
            img = cv2.imread(filename)[:, :, ::-1]
            st.image(img, use_column_width=True)

    st.markdown("### Ground Truth")
    col = st.columns(max_display_per_col)
    for i, article_id in enumerate(targets['gt']):
        j = i % max_display_per_col
        with col[j]:
            st.write(f"id:{article_id}")
            filename = str(image_dir + f'{article_id[:3]}/{article_id}.jpg')
            img = cv2.imread(filename)[:, :, ::-1]
            st.image(img, use_column_width=True)


def main():
    # config
    with open('./config.yaml', 'r') as yml:
        config = yaml.safe_load(yml)
    dataset = config['common']['dataset']

    data_dir = config["common"]['data_dir']
    image_dir = config["common"]['image_dir']
    min_purchased_count = config["customers"]['min_purchased_count']
    num_sample = config["customers"]['num_sample']
    max_display_per_col = config["customers"]['max_display_per_col']

    # read data(use cache to reduce loading time)
    df, articles, customers = read_data(data_dir)
    unique_customers, active_unique_customers, unique_articles = get_sub_data(df, customers, articles, min_purchased_count)

    # read prediction result
    pred_df = pd.read_pickle('../output/merged.pkl')
    users = pd.read_pickle(f"../input/{dataset}/users.pkl")
    items = pd.read_pickle(f"../input/{dataset}/items.pkl")

    pred_df = pred_df.merge(users[['user', 'customer_id']], on='user').drop('user', axis=1)
    item_id_map = items[['item', 'article_id']].to_dict()['article_id']
    pred_df['gt'] = pred_df['gt'].apply(lambda x: [item_id_map[y] for y in x])
    pred_df['item'] = pred_df['item'].apply(lambda x: [item_id_map[y] for y in x])

    # select type
    analysis_type = st.sidebar.radio("Select analysis type", ["Customers", "Articles", "Precitions"])

    # visualize
    if analysis_type == "Customers":
        visualize_customer(df, customers, unique_customers, active_unique_customers, num_sample, max_display_per_col, image_dir)
    elif analysis_type == "Articles":
        visualize_article(articles, unique_articles, image_dir)
    elif analysis_type == "Precitions":
        visualize_prediction(pred_df, customers, unique_customers, max_display_per_col, image_dir)
    else:
        NotImplementedError


if __name__ == "__main__":
    main()
