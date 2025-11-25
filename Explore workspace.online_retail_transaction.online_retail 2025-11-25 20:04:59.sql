-- Databricks notebook source
-- MAGIC %md
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
