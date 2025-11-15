# Savills Office Space Analysis by Industry

**Language:** Python  
**Libraries:** pandas, numpy, matplotlib, seaborn, sklearn, graphviz  
**Dataset:** Savills Leases & Market Data (DataFest 2025) — office lease data across multiple U.S. markets  
**Group Project:** Completed as part of a collaborative team effort.

---

## Objective

1. Aid Savills in identifying the **best office space for clients by industry**  
2. Visualize the **impact of COVID-19** on office occupancy for top industries  

---

## Data Description

- Covers **office leases across multiple U.S. markets**  
- **Key Variables:**
  - `internal_industry`: Industry type of the tenant  
  - `market`: Market region containing several industries  
  - `leasedSF`: Square footage of leased office space  
  - `CBD_suburban`: Location classification (CBD vs. Suburban)  
  - `zip`: Zip code of the leased office  
  - `state`: U.S. state  
  - `overall_rent`: Average rent in the market per quarter  

**Data cleaning and preparation steps:**
1. Removed all observations with missing `internal_industry`  
2. Converted categorical variables (`internal_industry`, `state`, `CBD_suburban`) to numeric codes  
3. Selected **top four prominent industries** for analysis: Legal, Business & Services, Technology, Finance & Insurance  

---

## Exploratory Data Analysis

- Visualized **industry distribution by state and market** using `seaborn.countplot`  
- Examined correlations among `leasedSF`, `available_space`, and `overall_rent` using heatmaps  

**Key Observations:**
- CBD locations dominate across all industries  
- NYC consistently appears as a **top location** for all industries  
- Legal Services prefer **small to medium** offices  
- Business Services tend to occupy **smaller** spaces  
- Finance & Insurance and Technology tend to occupy **larger** spaces  

**Visuals:**
- Countplot of industries by state  
- Countplot of industries by market  
- Heatmap of key office space metrics  

---

## Part 1: Predicting Best Office Space for a Client

**Goal:** Determine which office spaces are most suitable for each industry using **Random Forest classification**  

**Predictors:** `leasedSF`, `zip`, `CBD_suburban`, `state`  
**Targets:** `internal_industry` (multi-class: Legal, Business, Technology, Finance & Insurance)  

**Random Forest Feature Importance:**

| Industry        | Top Features (Importance) |
|-----------------|---------------------------|
| Legal           | Leased SF (0.569), Zip Code (0.200) |
| Finance & Insurance | Leased SF (0.513), Zip Code (0.213) |
| Technology      | Leased SF (0.517), Zip Code (0.253) |
| Business & Services | Leased SF (0.537), Zip Code (0.253) |

**Insights:**
- **Square footage** and **zip code** are the most influential factors across all four industries  
- **Business Services:** prefer smaller office spaces  
- **Finance & Insurance** and **Technology:** favor larger spaces  
- **Legal Services:** prefer small to medium offices  

**Top Zip Codes by Industry:**

| Industry | Top Zip Codes |
|----------|---------------|
| Legal | 100 (NYC), 200 (Washington DC), 606 (Chicago), 770 (Houston), 900 (Los Angeles) |
| Finance & Insurance | 100 (NYC), 101 (NYC), 606 (Chicago), 303 (Atlanta), 770 (Houston) |
| Technology | 100 (NYC), 941 (San Francisco), 606 (Chicago), 201 (New Jersey), 787 (Austin) |
| Business & Services | 100 (NYC), 201 (New Jersey), 200 (Washington DC), 606 (Chicago), 222 (North Virginia) |

**Visuals:**
- Random Forest example tree  
- Feature importance plots  

---

## Part 2: Visualizing COVID-19 Impact on Office Occupancy

**Goal:** Examine how COVID-19 affected top four industries in Manhattan  

**Findings:**
- **Finance & Insurance:** Consistently led Manhattan market, starting at 21.03% (Q1 2018) → 22.5% (Q4 2024). Spiked to 29.17% in Q2 2020 as other industries scaled down.  
- **Business Services:** Remained under 10%, hitting a low of 1.96% during 2020 Q2, likely due to shutdowns.  
- **Legal Services:** Declined slightly during COVID, slower adaptation compared to finance.  
- **Technology:** Maintained a steady presence, adapted well to remote work and virtual services.  

**Visuals:**
- Line plots of industry market share over time by quarter  
- Highlighted COVID-19 impact period (Q2 2020)

---

## Key Findings

- **Best Office Space Selection:**  
  - Square footage and zip code are strongest determinants  
  - Industry-specific recommendations:  
    - Business Services → smaller spaces  
    - Finance & Insurance, Technology → larger spaces  
    - Legal → small to medium spaces  
- **COVID-19 Insights:**  
  - Finance & Insurance resilient, quickly adapted to remote/virtual needs  
  - Business Services and Legal experienced notable declines during early COVID  
  - NYC remains dominant across all industries, but oversaturation may occur  


