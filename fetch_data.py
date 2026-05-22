import yfinance as yf
import pandas as pd

print("--- Yahoo Finance se Data Fetch ho raha hai... ---")

# Is baar Reliance Industries limited ka ticker use kar rahe hain
ticker = "RELIANCE.NS"

# Pichle 5 saal ka data download karenge period parameters ke sath (Ye jyada safe hai)
stock_data = yf.download(ticker, period="5y")

# Dekhte hain data aaya ya nahi
print("\nData successfully fetched! Shuruat ke 5 rows ye hain:")
print(stock_data.head())

# Data ko check karte hain ki khali to nahi hai
if not stock_data.empty:
    # Data ko CSV me save karenge
    stock_data.to_csv("reliance_data.csv")
    print("\nData ko 'reliance_data.csv' me save kar diya gaya hai! Balle Balle!")
else:
    print("\nAbhi bhi DataFrame khali hai, check internet connection.")