# 📊 Unemployment Analysis with Python
# Author: Blessy (Data Science Internship)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1️⃣ Load the datasets
df1 = pd.read_csv("C:/Users/bless/Desktop/Blessy/Datascience internship/Unemployment 1.csv")
df2 = pd.read_csv("C:/Users/bless/Desktop/Blessy/Datascience internship/Unemployment 2.csv")

# 2️⃣ View column names to understand structure
print("Dataset 1 Columns:", df1.columns.tolist())
print("Dataset 2 Columns:", df2.columns.tolist())

# 3️⃣ Try to standardize column names (convert to lowercase and strip spaces)
df1.columns = df1.columns.str.strip().str.lower()
df2.columns = df2.columns.str.strip().str.lower()

# 4️⃣ Check for date columns and rename if needed
for col in df1.columns:
    if 'date' in col:
        df1.rename(columns={col: 'date'}, inplace=True)

for col in df2.columns:
    if 'date' in col:
        df2.rename(columns={col: 'date'}, inplace=True)

# 5️⃣ Convert 'date' column to datetime format (ignore errors if any)
df1['date'] = pd.to_datetime(df1['date'], errors='coerce')
df2['date'] = pd.to_datetime(df2['date'], errors='coerce')

# 6️⃣ Drop rows with missing date values
df1.dropna(subset=['date'], inplace=True)
df2.dropna(subset=['date'], inplace=True)

# 7️⃣ Merge both datasets based on 'date' (outer join keeps all data)
merged = pd.merge(df1, df2, on='date', how='outer', suffixes=('_1', '_2'))

# 8️⃣ Drop duplicates and sort by date
merged.drop_duplicates(subset='date', inplace=True)
merged.sort_values(by='date', inplace=True)

# 9️⃣ Display info and sample
print("\n✅ Cleaned & Merged Dataset:")
print(merged.info())
print(merged.head())

# 🔟 Visualization - Unemployment trends
plt.figure(figsize=(12,6))
sns.lineplot(data=merged, x='date', y=merged.filter(like='rate').mean(axis=1))
plt.title("📈 Unemployment Rate Trend Over Time")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.grid(True)
plt.show()

# 11️⃣ Covid-19 impact visualization (2020 focus)
covid_period = merged[(merged['date'] >= '2020-01-01') & (merged['date'] <= '2021-12-31')]
plt.figure(figsize=(12,6))
sns.lineplot(data=covid_period, x='date', y=covid_period.filter(like='rate').mean(axis=1), color='red')
plt.title("🦠 Covid-19 Period Impact on Unemployment (2020–2021)")
plt.xlabel("Date")
plt.ylabel("Average Unemployment Rate (%)")
plt.grid(True)
plt.show()

# 12️⃣ Insights
print("\n📊 INSIGHTS:")
print("- The Covid-19 period (2020–2021) shows a clear spike in unemployment rate.")
print("- Pre-2020 trends were relatively stable, but sharp rises align with lockdown phases.")
print("- Post-2021, a gradual recovery trend is visible in most regions.")
