import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from babel.numbers import format_currency
import geopandas as gpd
import folium
from streamlit_folium import folium_static
sns.set(style='dark')

def create_monthly_orders_df(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["year_month"] = df["order_purchase_timestamp"].dt.to_period("M")

    monthly_orders_df = df.groupby("year_month").agg({
        "order_id": "nunique",  
        "price": "sum",       
        "freight_value": "sum" 
    }).reset_index()

    monthly_orders_df["year_month"] = monthly_orders_df["year_month"].dt.to_timestamp()

    monthly_orders_df.rename(columns={
        "order_id": "order_count",
        "price": "revenue"
    }, inplace=True)

    return monthly_orders_df

def create_rfm_df(df):
    df["total_price"] = df["price"] + df["freight_value"]
    
    rfm_df = df.groupby(by="customer_unique_id", as_index=False).agg({
        "order_purchase_timestamp": "max", 
        "order_id": "nunique", 
        "total_price": "sum"  
    })
    
    rfm_df.rename(columns={
        "order_purchase_timestamp": "max_order_timestamp",
        "order_id": "frequency",
        "total_price": "monetary"
    }, inplace=True)
    
    rfm_df["max_order_timestamp"] = pd.to_datetime(rfm_df["max_order_timestamp"]).dt.date
    recent_date = df["order_purchase_timestamp"].max().date()    
    rfm_df["recency"] = rfm_df["max_order_timestamp"].apply(lambda x: (recent_date - x).days)   
    rfm_df.drop("max_order_timestamp", axis=1, inplace=True)
    
    return rfm_df

def create_category_sales_df(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    last_6_months = df["order_purchase_timestamp"].max() - pd.DateOffset(months=6)
    filtered_df = df[df["order_purchase_timestamp"] >= last_6_months]

    category_sales_df = filtered_df.groupby("product_category_name").agg({
        "order_id": "nunique" 
    }).reset_index()

    category_sales_df.rename(columns={"order_id": "sales_count"}, inplace=True)

    return category_sales_df

def create_payment_distribution_df(df):
    payment_counts = df["payment_type"].value_counts().reset_index()
    payment_counts.columns = ["payment_type", "transaction_count"]
    payment_counts["percentage"] = (payment_counts["transaction_count"] / payment_counts["transaction_count"].sum()) * 100
    payment_counts = payment_counts[payment_counts["percentage"] >= 2]

    return payment_counts

def create_delivery_time_df(df):
    df["order_purchase_timestamp"] = pd.to_datetime(df["order_purchase_timestamp"])
    df["order_delivered_customer_date"] = pd.to_datetime(df["order_delivered_customer_date"])
    df["delivery_time_days"] = (df["order_delivered_customer_date"] - df["order_purchase_timestamp"]).dt.days
    df_clean = df.dropna(subset=["delivery_time_days"])

    return df_clean

def create_review_distribution_df(df):
    rating_counts = df["review_score"].value_counts().sort_index()
    rating_percentage = (rating_counts / rating_counts.sum()) * 100
    df_ratings = rating_counts.reset_index()
    df_ratings.columns = ["Review Score", "Review Count"]

    return df_ratings, rating_percentage

all_df = pd.read_csv("main_data.csv")
datetime_columns = ["order_purchase_timestamp", "order_delivered_customer_date"]
all_df.sort_values(by="order_purchase_timestamp", inplace=True)
all_df.reset_index(drop=True, inplace=True)

for column in datetime_columns:
    all_df[column] = pd.to_datetime(all_df[column])

min_date = all_df["order_purchase_timestamp"].min().date()
max_date = all_df["order_purchase_timestamp"].max().date()

with st.sidebar:
    st.image("../logo.webp")
    start_date, end_date = st.date_input(
        label='Rentang Waktu',
        min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

main_df = all_df[(all_df["order_purchase_timestamp"] >= str(start_date)) & 
                (all_df["order_purchase_timestamp"] <= str(end_date))]

monthly_orders_df = create_monthly_orders_df(main_df)
category_sales_df = create_category_sales_df(main_df)
payment_counts = create_payment_distribution_df(main_df)
orders_clean = create_delivery_time_df(main_df)
df_ratings, rating_percentage = create_review_distribution_df(main_df)
rfm_df = create_rfm_df(main_df)

st.header('Garie Public E-Commerce :sparkles:')

st.subheader('Monthly Orders')
col1, col2 = st.columns(2)

with col1:
    total_orders = monthly_orders_df["order_count"].sum()
    st.metric("Total Orders", value=total_orders)

with col2:
    total_revenue = format_currency(
        monthly_orders_df["revenue"].sum() + monthly_orders_df["freight_value"].sum(), "IDR", locale='id_ID'
    )
    st.metric("Total Revenue", value=total_revenue)

fig, ax = plt.subplots(figsize=(16, 8))

ax.plot(
    monthly_orders_df["year_month"],
    monthly_orders_df["order_count"],
    marker='o',
    linewidth=2,
    color="#90CAF9"
)

ax.tick_params(axis='y', labelsize=20)
ax.tick_params(axis='x', labelsize=15)
plt.xticks(rotation=45)

st.pyplot(fig)
top_categories = category_sales_df.nlargest(5, "sales_count")
bottom_categories = category_sales_df.nsmallest(5, "sales_count")

st.subheader("Product Sales Performance (Last 6 Months)")

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16,6))

top_colors = ['#a0c4ff'] * len(top_categories)
top_colors[0] = '#1f77b4'  

bottom_colors = ['#ffb3b3'] * len(bottom_categories)
bottom_colors[0] = '#d62728'  

sns.barplot(data=top_categories,
            x="sales_count",
            y="product_category_name",
            ax=axes[0],
            palette=top_colors)

axes[0].set_title("Top 5 Best-Selling Product Categories", fontsize=14, fontweight='bold')
axes[0].set_xlabel("Number of Units Sold", fontsize=12)
axes[0].set_ylabel("") 
axes[0].tick_params(axis='y', labelsize=10)
axes[0].tick_params(axis='x', labelsize=10)

sns.barplot(data=bottom_categories,
            x="sales_count",
            y="product_category_name",
            ax=axes[1],
            palette=bottom_colors)

axes[1].set_title("Top 5 Lowest-Selling Product Categories", fontsize=14, fontweight='bold')
axes[1].set_xlabel("Number of Units Sold", fontsize=12)
axes[1].set_ylabel("")  
axes[1].tick_params(axis='y', labelsize=10)
axes[1].tick_params(axis='x', labelsize=10)

for ax in axes:
    ax.grid(axis='x', linestyle='--', alpha=0.7)

plt.tight_layout()

st.pyplot(fig)

st.subheader("Payment Method Distribution")

colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854"]

fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    payment_counts["transaction_count"],
    labels=payment_counts["payment_type"],
    autopct="%1.1f%%",
    startangle=140,
    colors=colors,
    wedgeprops={"edgecolor": "black", "linewidth": 1},
    textprops={"fontsize": 12, "weight": "bold"},
    pctdistance=0.85  
)

centre_circle = plt.Circle((0, 0), 0.70, fc="white")
fig.gca().add_artist(centre_circle)

plt.title("Distribution of Payment Methods", fontsize=14, fontweight="bold", color="#333")

for autotext in autotexts:
    autotext.set_color("white")
    autotext.set_weight("bold")

st.pyplot(fig)

average_delivery_time = orders_clean["delivery_time_days"].mean()
max_delivery_time = orders_clean["delivery_time_days"].quantile(0.95)

st.subheader("Average Delivery Time")

fig, ax = plt.subplots(figsize=(8, 3))
ax.barh(["Average Delivery Time"], [average_delivery_time], color="#69b3a2", height=0.4)
ax.text(average_delivery_time - 1, 0, f"{average_delivery_time:.2f} days",
        va="center", ha="right", fontsize=12, fontweight="bold", color="white")
ax.axvline(x=max_delivery_time, color="red", linestyle="--", linewidth=1.5, label=f"Max Threshold ({max_delivery_time:.2f} days)")
ax.set_xlabel("Days")
ax.set_title("Average Delivery Time")
ax.legend()

st.pyplot(fig)

colors = ["#3498db" if score != 5 else "#2ecc71" for score in df_ratings["Review Score"]]  

st.subheader("Review Rating Distribution")

fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=df_ratings["Review Score"], y=df_ratings["Review Count"], palette=colors, ax=ax)

for i, (value, percent) in enumerate(zip(df_ratings["Review Count"], rating_percentage)):
    ax.text(i, value + 1000, f"{value:,}\n({percent:.1f}%)", ha="center", fontsize=11, fontweight="bold")

ax.set_title("Review Rating Distribution", fontsize=14, fontweight="bold")
ax.set_xlabel("Review Score", fontsize=12)
ax.set_ylabel("Review Count", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)

st.pyplot(fig)

st.subheader("Relationship Between Product Price and Units Sold")

if 'price' in main_df.columns and 'order_id' in main_df.columns:
    product_sales = main_df.groupby('product_id').agg(
        total_revenue=('price', 'sum'),
        units_sold=('order_id', 'count')
    ).reset_index()

    product_sales['avg_price'] = product_sales['total_revenue'] / product_sales['units_sold']
    fig, ax = plt.subplots(figsize=(8, 5))

    sns.regplot(x=product_sales['avg_price'],
                y=product_sales['units_sold'],
                scatter_kws={'alpha': 0.3, 'color': '#1f77b4'},
                line_kws={'color': 'red'},
                ax=ax)

    ax.set_xscale('log')
    ax.set_title('Relationship Between Product Price and Units Sold', fontsize=14, fontweight='bold')
    ax.set_xlabel('Average Product Price (Log Scale)', fontsize=12)
    ax.set_ylabel('Units Sold', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)

    correlation_value = product_sales[['avg_price', 'units_sold']].corr().iloc[0, 1]

    ax.text(ax.get_xlim()[1] * 0.5, ax.get_ylim()[1] * 0.9,
            f'Correlation: {correlation_value:.2f}',
            fontsize=14, color='black', bbox=dict(facecolor='white', alpha=0.6))
    ax.set_ylim(0, max(product_sales['units_sold']) * 1.1)

    st.pyplot(fig)

st.subheader('Top 10 Cities with the Most Customers')

customer_geo_distribution = main_df.groupby('customer_city').size().reset_index(name='total_customers')
top_cities = customer_geo_distribution.sort_values(by='total_customers', ascending=False).head(10)
colors = ['#1f4e79' if i == 0 else '#90c3d4' for i in range(len(top_cities))]
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(top_cities['customer_city'], top_cities['total_customers'], color=colors)

for bar, value in zip(bars, top_cities['total_customers']):
    ax.text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2, str(value),
            va='center', fontsize=10, fontweight='bold', color='black')

ax.set_xlabel("Number of Customers", fontsize=12)
ax.set_ylabel("City", fontsize=12)
ax.set_title("Top 10 Cities with the Most Customers", fontsize=14, fontweight='bold')
ax.invert_yaxis()

st.pyplot(fig)

st.subheader("🏆 Best Customers Based on RFM Parameters")

required_columns = {'customer_unique_id', 'recency', 'frequency', 'monetary'}
if required_columns.issubset(rfm_df.columns):

    col1, col2, col3 = st.columns(3)

    with col1:
        avg_recency = round(rfm_df['recency'].mean(), 1)
        st.metric("📅 Average Recency (days)", value=avg_recency)

    with col2:
        avg_frequency = round(rfm_df['frequency'].mean(), 2)
        st.metric("🔁 Average Frequency", value=avg_frequency)

    with col3:
        avg_monetary = format_currency(rfm_df['monetary'].mean(), "AUD", locale='es_CO')
        st.metric("💰 Average Monetary", value=avg_monetary)

    rfm_df["customer_unique_id"] = rfm_df["customer_unique_id"].astype(str)

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    def get_colors(df, column, dark_color="#1976D2", light_color="#90CAF9"):
        colors = [light_color] * len(df)
        colors[0] = dark_color 
        return colors

    top_recency = rfm_df.sort_values(by="recency", ascending=True).head(5)
    sns.barplot(y="recency", x="customer_unique_id", data=top_recency,
                palette=get_colors(top_recency, "recency"), ax=ax[0])
    ax[0].set_ylabel(None)
    ax[0].set_xlabel("Customer Unique ID", fontsize=12)
    ax[0].set_title("By Recency (days)", fontsize=14, fontweight='bold')
    ax[0].tick_params(axis='x', rotation=45)

    top_frequency = rfm_df.sort_values(by="frequency", ascending=False).head(5)
    sns.barplot(y="frequency", x="customer_unique_id", data=top_frequency,
                palette=get_colors(top_frequency, "frequency"), ax=ax[1])
    ax[1].set_ylabel(None)
    ax[1].set_xlabel("Customer Unique ID", fontsize=12)
    ax[1].set_title("By Frequency", fontsize=14, fontweight='bold')
    ax[1].tick_params(axis='x', rotation=45)

    top_monetary = rfm_df.sort_values(by="monetary", ascending=False).head(5)
    sns.barplot(y="monetary", x="customer_unique_id", data=top_monetary,
                palette=get_colors(top_monetary, "monetary"), ax=ax[2])
    ax[2].set_ylabel(None)
    ax[2].set_xlabel("Customer Unique ID", fontsize=12)
    ax[2].set_title("By Monetary", fontsize=14, fontweight='bold')
    ax[2].tick_params(axis='x', rotation=45)

    st.pyplot(fig)

else:
    st.warning("Dataset tidak memiliki kolom yang dibutuhkan! Pastikan dataset memuat kolom 'customer_unique_id', 'recency', 'frequency', dan 'monetary'.")

geo_df = pd.read_csv("../data/geolocation_dataset.csv")
geo_df["geolocation_lat"] = pd.to_numeric(geo_df["geolocation_lat"], errors='coerce')
geo_df["geolocation_lng"] = pd.to_numeric(geo_df["geolocation_lng"], errors='coerce')
geo_df = geo_df.groupby("geolocation_zip_code_prefix").agg({
    "geolocation_lat": "mean",
    "geolocation_lng": "mean"
}).reset_index()

main_df = main_df.merge(geo_df, left_on="customer_zip_code_prefix", right_on="geolocation_zip_code_prefix", how="left")

city_counts = main_df["customer_city"].value_counts().reset_index()
city_counts.columns = ["customer_city", "customer_count"]

main_df = main_df.merge(city_counts, on="customer_city", how="left")

def create_customer_map(df):
    m = folium.Map(location=[-2.5489, 118.0149], zoom_start=5)
    
    for _, row in df.iterrows():
        if not np.isnan(row["geolocation_lat"]) and not np.isnan(row["geolocation_lng"]):
            folium.CircleMarker(
                location=[row["geolocation_lat"], row["geolocation_lng"]],
                radius=row["customer_count"] / 50,  
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.5,
                popup=f"{row['customer_city']}: {row['customer_count']} customers"
            ).add_to(m)
    
    return m

def apply_manual_clustering(rfm_df):
    def safe_qcut(series, q, labels):
        unique_vals = series.nunique()
        if unique_vals < q:
            return pd.cut(series, bins=unique_vals, labels=labels[:unique_vals], include_lowest=True)
        try:
            return pd.qcut(series, q=q, labels=labels, duplicates='drop')
        except ValueError:
            return pd.cut(series, bins=q, labels=labels[:q], include_lowest=True)
    
    rfm_df['R_Score'] = safe_qcut(rfm_df['recency'], q=3, labels=[3, 2, 1])
    rfm_df['F_Score'] = safe_qcut(rfm_df['frequency'], q=3, labels=[1, 2, 3])
    rfm_df['M_Score'] = safe_qcut(rfm_df['monetary'], q=3, labels=[1, 2, 3])
    rfm_df['Cluster'] = rfm_df[['R_Score', 'F_Score', 'M_Score']].sum(axis=1)
    
    return rfm_df
st.subheader("Geospatial Analysis: Customer Distribution Map")
customer_map = create_customer_map(main_df)
folium_static(customer_map)

st.subheader("Customer Clustering based on RFM Scores")
rfm_clustered = apply_manual_clustering(rfm_df)
st.dataframe(rfm_clustered[['customer_unique_id', 'recency', 'frequency', 'monetary', 'Cluster']])

fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(rfm_clustered['Cluster'], bins=6, kde=True, color='blue', ax=ax)
ax.set_title("Distribution of Customer Clusters", fontsize=14, fontweight='bold')
ax.set_xlabel("Cluster Group")
ax.set_ylabel("Number of Customers")
st.pyplot(fig)

st.caption('Copyright © Gigih Agung Prasetyo 2025')