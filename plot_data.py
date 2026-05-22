import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("--- 'reliance_data.csv' se data read ho raha hai... ---")

# CSV file ko read karenge aur Date column ka format sahi karenge
# index_col=[0] ka matlab pehla column (Date) hamara index banega
df = pd.read_csv("reliance_data.csv", header=[0, 1], index_col=[0])

# Column ke naye multi-level names ko simple karte hain
df.columns = df.columns.get_level_values(0)

print("\nData sahi se read ho gaya hai! Rows aur Columns ka count:", df.shape)

# Chart ka size bade aur saaf format me set karte hain
plt.figure(figsize=(12, 6))

# Hum 'Close' price (yaani jab market band hua tab ka price) ka graph banayenge
plt.plot(df.index, df['Close'], label='Reliance Stock Price', color='blue', linewidth=2)

# Chart ke upar labels lagate hain
plt.title('Reliance Industries - 5 Year Stock Price Trend', fontsize=16)
plt.xlabel('Date (Year)', fontsize=12)
plt.ylabel('Stock Price (INR)', fontsize=12)
plt.grid(True) # Background me grid lines lagane ke liye
plt.legend()

# Chart ko screen par show karne ke liye
print("\nGraph open ho raha hai... (Aapki screen par ek alag window khulegi)")
plt.show()