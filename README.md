# Marketing Performance Dashboard

## 📊 Project Overview

The **Marketing Performance Dashboard** is a Data Analytics Capstone
Project developed for SkillOrbit. The project analyzes customer
marketing data to understand customer demographics, spending behaviour,
purchase channels, campaign acceptance, latest campaign response, and
customer segments.

An interactive dashboard is developed using **Streamlit**, with data
processing and analysis performed using **Python, Pandas, NumPy, Plotly,
and Scikit-learn**.

------------------------------------------------------------------------

## 🎯 Project Objectives

- Load and inspect the marketing campaign dataset.
- Clean missing values, duplicates, invalid birth-year records, and
  extreme income values.
- Create analytical features such as Age, Total Spending, Total
  Purchases, and Total Campaign Acceptance.
- Perform exploratory data analysis on customer demographics, income,
  spending, products, purchase channels, and campaign response.
- Calculate meaningful marketing and customer KPIs available in the
  dataset.
- Apply **K-Means clustering** for customer segmentation.
- Build an interactive Streamlit dashboard with filters and
  visualizations.
- Generate data-driven business insights and recommendations.

------------------------------------------------------------------------

## 📁 Project Structure

``` text
Marketing-Performance-Dashboard/
│
├── appp.py
├── requirements.txt
├── README.md
│
├── data/
│   └── marketing_campaign.csv
│
├── notebooks/
│   └── market_analysis.ipynb
│
├── report/
│   └── Marketing_Performance_Dashboard_Project_Report.docx
│
├── ppt/
│   └── Marketing_Performance_Dashboard.pptx
│
└── screenshots/
    ├── dashboard_overview.png
    ├── segmentation.png
    └── business_insights.png
```

> `app(1).ipynb` is not required for running the dashboard. The main
> dashboard application is `appp.py`.

------------------------------------------------------------------------

## 📊 Dataset

The supplied marketing campaign dataset initially contains:

- **2,240 customer records**
- **29 columns**

After the documented cleaning process, the analytical dataset contains
**2,236 customers**.

### Dataset Source

[Kaggle – Marketing Campaign
Dataset](https://www.kaggle.com/datasets/techstarmahesh/marketing-compaign)

The dataset contains customer demographic, purchasing, product spending,
web activity, and campaign-response information.

------------------------------------------------------------------------

## 🧹 Data Cleaning & Preprocessing

The project applies the following cleaning steps:

- Converted `Dt_Customer` to datetime.
- Checked and handled missing `Income` values using median imputation.
- Checked duplicate records.
- Removed unrealistic `Year_Birth` values below 1920.
- Removed the documented extreme income value above \$200,000.
- Calculated customer age using **2014** as the reference year.
- Created additional analytical features.

### Feature Engineering

The following features are created:

- `Age`
- `Total_Spending`
- `Total_Purchases`
- `Total_Campaign_Accepted`
- `Total_Children`

------------------------------------------------------------------------

## 📌 Key Performance Indicators

The dashboard calculates KPIs that are directly supported by the
dataset:

| KPI                      |       Value |
|--------------------------|------------:|
| Total Customers          |       2,236 |
| Average Income           |    \$51,953 |
| Average Spending         |       \$606 |
| Total Spending           | \$1,354,986 |
| Average Purchases        |        12.5 |
| Campaign Response Rate   |      14.94% |
| Campaign Acceptance Rate |       5.96% |
| Complaint Rate           |       0.89% |
| Average Recency          |   49.1 days |

> The dataset does not contain advertising impressions, ad clicks,
> advertising cost, reach, or advertising revenue. Therefore,
> traditional advertising CTR, advertising conversion rate, and ROI are
> not calculated.

------------------------------------------------------------------------

## 👥 Customer Segmentation

Customer segmentation is performed using **K-Means clustering**.

### Clustering Features

- Income
- Recency
- Total Spending
- Web Purchases
- Catalog Purchases
- Store Purchases
- Total Campaign Acceptance

The features are standardized using `StandardScaler`.

Candidate values from **K = 2 to K = 8** are evaluated using silhouette
scores.

The final implementation selects **K = 2**, which produced the highest
silhouette score among the tested values.

------------------------------------------------------------------------

## 📈 Dashboard Features

The Streamlit dashboard contains the following sections:

### Overview

- KPI cards
- Education distribution
- Marital-status distribution
- Age distribution
- Income distribution

### Customer Analysis

- Income vs. spending
- Age analysis
- Education-based analysis
- Customer engagement analysis

### Spending & Channels

- Product-category spending
- Web/Catalog/Store purchase comparison
- Product spending distribution
- Purchase behaviour

### Campaign Performance

- Previous campaign acceptance
- Campaign acceptance rates
- Latest campaign response
- Response by education

### Customer Segmentation

- K-Means segment distribution
- Segment profiles
- Segment spending
- Segment purchases
- Segment response

### Business Insights

- Highest spending product category
- Highest purchase channel
- Campaign acceptance observations
- Segment-level spending and response observations
- Engagement relationships

### Data Explorer

- Filtered customer-level data
- Selectable columns
- CSV download

------------------------------------------------------------------------

## 🛠️ Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Plotly**
- **Scikit-learn**
- **Streamlit**
- **Jupyter Notebook**

------------------------------------------------------------------------

## ⚙️ Installation

Clone the repository:

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Marketing-Performance-Dashboard
```

Install the required Python packages:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## ▶️ Run the Dashboard

Make sure the dataset is available at:

``` text
data/marketing_campaign.csv
```

Then run:

``` bash
python -m streamlit run appp.py
```

The Streamlit dashboard will open in your browser.

------------------------------------------------------------------------

## 📂 Expected Dataset Format

The application expects the marketing dataset as a semicolon-separated
CSV file:

``` text
marketing_campaign.csv
```

The application uses:

``` python
pd.read_csv(..., sep=";")
```

------------------------------------------------------------------------

## 🔍 Important Interpretation Notes

- `Web`, `Catalog`, and `Store` represent **purchase channels**, not
  digital advertising channels.
- `Response` represents the **latest campaign response available in the
  dataset** and is not treated as a true advertising sales-conversion
  rate.
- Correlation results describe association and do not establish
  causation.
- K-Means segment labels are model-generated labels and should be
  interpreted using their measured customer profiles.

------------------------------------------------------------------------

## 📌 Project Deliverables

- Source code
- Analytical Jupyter Notebook
- Dataset
- Interactive Streamlit dashboard
- Project report
- Project presentation
- Dashboard screenshots
- Requirements file
- GitHub repository

------------------------------------------------------------------------

## 👨‍💻 Team

**Prakhar Singh**  
**Suneel Mahor**

------------------------------------------------------------------------

## 📚 References

1.  SkillOrbit Data Analytics Capstone Project brief – Marketing
    Performance Dashboard.
2.  Supplied marketing campaign customer dataset.
3.  Project analytical notebook.
4.  Streamlit dashboard source code.

------------------------------------------------------------------------

## 🚀 Future Scope

- Integrate advertising data containing impressions, clicks, ad spend,
  reach, and revenue.
- Add time-based campaign analysis when campaign dates are available.
- Connect the dashboard to a database for larger datasets.
- Add automated report generation.
- Evaluate additional clustering methods.
- Deploy the Streamlit dashboard using a cloud hosting platform.

------------------------------------------------------------------------

## 🔗 Project Links

**Deployed Link:** 
https://marketing-performance-dashboard-deployment.streamlit.app/

**Dataset:**  
https://www.kaggle.com/datasets/techstarmahesh/marketing-compaign

**GitHub Repository:**  
https://github.com/prakharsinghhhh/Marketing-Performance-Dashboard
