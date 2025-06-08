import pandas as pd

# Read Excel file
df = pd.read_excel("./content/Classification.xlsx")
# df = pd.read_csv("./content/data1.csv", encoding='utf-8')

# Save as CSV with semicolon as separator
df.to_csv('./content/data.csv', index=False, header=True, sep=';', encoding='utf-8')

# Print the first few rows to verify
print(df.head())
