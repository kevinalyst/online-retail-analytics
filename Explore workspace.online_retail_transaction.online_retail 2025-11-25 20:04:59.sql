-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Data Preparation
-- MAGIC Right after ingest the table into schema, I notice the InvoiceDate column is in a wrong datetype, tried to fix it by using CAST() and TRY_CAST(), but it just throw null value. Instead of wrestling with whatever weirdness the column is doing, just rebuild the string into ISO format and cast it. That’s bulletproof.

-- COMMAND ----------

-- Fix datatype anomiles in InvoiceDate, create stage table
CREATE OR REPLACE TABLE online_retail_transaction.online_retail_stage AS (
  SELECT 
    InvoiceNo AS invoice_number,
    StockCode AS item_code,
    Description AS item_description,
    Quantity AS quantity,
    to_date(
      concat(
      substring(trim(InvoiceDate), 7, 4),  '-',  -- year
      substring(trim(InvoiceDate), 4, 2),  '-',  -- month
      substring(trim(InvoiceDate), 1, 2),  ''  -- day
      ) 
    ) AS invoice_date,
    to_timestamp(
      concat(
      substring(trim(InvoiceDate), 7, 4),  '-', 
      substring(trim(InvoiceDate), 4, 2),  '-', 
      substring(trim(InvoiceDate), 1, 2),  ' ',
      substring(trim(InvoiceDate), 12, 5), ':00'
      )
    ) AS invoice_timestamp,   
    UnitPrice AS price,
    CustomerID AS customer_id,
    Country AS country
  FROM online_retail_transaction.online_retail
);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Data Understanding & Exploration

-- COMMAND ----------

-- Basic profile
SELECT 
  COUNT(*) AS row_count, 
  MIN(o.invoice_date) AS min_invoice_date, 
  MAX(o.invoice_date) AS max_invoice_date,
  COUNT(DISTINCT o.customer_id) AS customers,
  COUNT(DISTINCT o.item_code) AS items,
  COUNT(DISTINCT o.country) AS countries
FROM online_retail_transaction.online_retail_stage o;

-- COMMAND ----------

-- Revenue distribution & sanity checks
SELECT
  SUM(o.quantity * o.price) AS revenue,
  (SELECT
    COUNT(*)
  FROM online_retail_transaction.online_retail_stage
  WHERE quantity < 0 OR LOWER(invoice_number) LIKE 'C%'
  ) AS returns,
  (SELECT
    COUNT(*)
  FROM online_retail_transaction.online_retail_stage
  WHERE customer_id IS NULL
  ) AS invaild_customers
FROM online_retail_transaction.online_retail_stage o;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Data Cleaning & Modeling

-- COMMAND ----------

-- Data cleaning
CREATE OR REPLACE TABLE online_retail_transaction.online_retail_clean  AS (
  SELECT 
    invoice_number,
    item_code,
    item_description,
    quantity,
    invoice_date,
    invoice_timestamp,
    price,
    quantity * price AS revenue,
    customer_id,
    CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END AS is_anonymous,
    CASE WHEN invoice_number LIKE 'C%' THEN 1 ELSE 0 END AS is_return,
    country
  FROM online_retail_transaction.online_retail_stage
);


-- COMMAND ----------

-- Date dimension
CREATE OR REPLACE TABLE online_retail_transaction.date_dim AS (
  SELECT
    DISTINCT invoice_date
  FROM online_retail_transaction.online_retail_clean
);

-- Country dimension
CREATE OR REPLACE TABLE online_retail_transaction.country_dim AS (
  SELECT
    DISTINCT country
  FROM online_retail_transaction.online_retail_clean
);

--Product dimension
CREATE OR REPLACE TABLE online_retail_transaction.product_dim AS (
  SELECT
    DISTINCT item_code,
    item_description
  FROM online_retail_transaction.online_retail_clean
);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # RFM Segmentation

-- COMMAND ----------

-- Compute RFM metrics
CREATE OR REPLACE TABLE online_retail_transaction.rfm_raw AS
WITH base AS (
  SELECT
    customer_id,
    MAX(invoice_date) AS last_purchase,
    COUNT(DISTINCT invoice_number) AS frequency,
    SUM(CASE WHEN is_return=0 THEN revenue ELSE 0 END) AS monetary
  FROM online_retail_transaction.online_retail_clean
  WHERE is_anonymous = 0
  GROUP BY customer_id  
),
ref_date AS (
  SELECT
    DATE_ADD(MAX(invoice_date), 1) AS ref_date
  FROM online_retail_transaction.online_retail_clean
)

SELECT
  customer_id,
  DATEDIFF(ref_date, last_purchase) AS recency,
  frequency,
  ROUND(monetary,2) AS monetary
  FROM base
  CROSS JOIN ref_date;

-- COMMAND ----------

--Compute RFM quaetiles scoring
CREATE OR REPLACE TABLE online_retail_transaction.rfm_score AS 
SELECT
  customer_id,
  NTILE(4) OVER (ORDER BY recency DESC) AS r_score,
  NTILE(4) OVER (ORDER BY frequency DESC) AS f_score,
  NTILE(4) OVER (ORDER BY monetary DESC) AS m_score
FROM online_retail_transaction.rfm_raw;


-- COMMAND ----------

--Map score into segament labels
CREATE OR REPLACE TABLE online_retail_transaction.rfm_segment AS 
SELECT
  rs.customer_id,
  rr.recency,
  rr.frequency,
  rr.monetary,
  rs.r_score,
  rs.f_score,
  rs.m_score,
  CASE
    WHEN r_score >= 3 AND f_score >= 3 AND m_score >= 3 THEN 'Champions'
    WHEN r_score >= 3 AND f_score >= 3 AND m_score <  3 THEN 'Loyal'
    WHEN m_score = 4   AND r_score <  3                 THEN 'Big Spenders'
    WHEN r_score = 1   AND f_score >= 2 AND m_score >= 2 THEN 'At Risk'
    WHEN r_score = 1   AND f_score <= 2 AND m_score <= 2 THEN 'Hibernating'
    ELSE 'Others'
  END AS Segment
FROM online_retail_transaction.rfm_score rs
LEFT JOIN online_retail_transaction.rfm_raw rr
USING (customer_id);

-- COMMAND ----------

-- MAGIC %md
-- MAGIC # Cohort Retention Analysis

-- COMMAND ----------

-- Assign acquisition cohort per customer
CREATE OR REPLACE TABLE online_retail_transaction.cohort AS 
  SELECT
    customer_id,
    DATE_TRUNC('month', MIN(invoice_date)) AS cohort_month
  FROM online_retail_transaction.online_retail_clean
  GROUP BY customer_id;

-- COMMAND ----------

-- Build cohort-month activity table
CREATE OR REPLACE TABLE online_retail_transaction.cohort_activity AS
WITH orders AS (
SELECT 
  c.customer_id,
  ct.cohort_month,
  DATE_TRUNC('month', c.invoice_date) AS order_month
FROM online_retail_transaction.online_retail_clean c
JOIN online_retail_transaction.cohort ct
USING (customer_id)
WHERE c.is_anonymous = 0 AND c.is_return = 0
),
add_index AS (
  SELECT
    *,
    MONTHS_BETWEEN(order_month, cohort_month) AS index
  FROM orders
)

SELECT
  cohort_month,
  index,
  COUNT(DISTINCT customer_id) AS active_customers
FROM add_index
GROUP BY cohort_month, index
ORDER BY cohort_month, index;

-- COMMAND ----------

-- Convert to retention rates
CREATE OR REPLACE TABLE online_retail_transaction.cohort_retention AS
WITH cohort_size AS (
SELECT
    cohort_month,
    MAX(CASE WHEN index = 0 THEN active_customers END) AS cohort_size
  FROM online_retail_transaction.cohort_activity
  GROUP BY cohort_month
)
SELECT
  ca.cohort_month,
  ca.index,
  ca.active_customers,
  cs.cohort_size,
  ca.active_customers / cs.cohort_size AS retention_rate
FROM online_retail_transaction.cohort_activity ca
JOIN cohort_size cs
USING (cohort_month);
