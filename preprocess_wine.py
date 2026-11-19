
# ASSIGNMENT 1 - TIEN XU LY DU LIEU
# De tai: Phan loai ruou vang (Wine Classification)

import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. DOC DU LIEU
print("===== 1. DOC DU LIEU =====")

wine = load_wine()
df = pd.DataFrame(wine.data, columns=wine.feature_names)
df["target"] = wine.target

print("5 dong dau tien:")
print(df.head())

print("\nKich thuoc du lieu:", df.shape)

# Luu du lieu goc
df.to_csv("wine_raw.csv", index=False)


# 2. KHAM PHA DU LIEU
print("\n===== 2. KHAM PHA DU LIEU =====")

print("\nThong tin du lieu:")
df.info()

print("\nThong ke mo ta:")
print(df.describe())

print("\nSo gia tri thieu:")
print(df.isnull().sum())

print("\nSo dong trung lap:", df.duplicated().sum())


# 3. LAM SACH DU LIEU
print("\n===== 3. LAM SACH DU LIEU =====")

# Xoa dong trung lap
df = df.drop_duplicates().copy()

# Xu ly gia tri thieu bang trung vi
feature_cols = df.columns.drop("target")
df[feature_cols] = df[feature_cols].fillna(
    df[feature_cols].median()
)

print("So gia tri thieu sau xu ly:",
      df.isnull().sum().sum())


# 4. CHIA DU LIEU TRAIN / TEST
print("\n===== 4. CHIA TRAIN / TEST =====")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("So mau Train:", len(X_train))
print("So mau Test:", len(X_test))


# 5. XU LY GIA TRI NGOAI LE
print("\n===== 5. XU LY OUTLIERS =====")

# Dung IQR va gioi han gia tri ngoai le
# Chi tinh nguong tu tap Train de tranh ro ri du lieu

Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = ((X_train < lower) | (X_train > upper))

print("So gia tri ngoai le trong Train:",
      outliers.sum().sum())

X_train = X_train.clip(lower=lower, upper=upper, axis=1)
X_test = X_test.clip(lower=lower, upper=upper, axis=1)

print("Da xu ly outliers bang phuong phap IQR")


# 6. CHUAN HOA DU LIEU
print("\n===== 6. CHUAN HOA DU LIEU =====")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X.columns,
    index=X_train.index
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X.columns,
    index=X_test.index
)

print("Da chuan hoa du lieu bang StandardScaler")


# 7. LUU DU LIEU SAU TIEN XU LY
print("\n===== 7. LUU DU LIEU =====")

train_processed = X_train_scaled.copy()
train_processed["target"] = y_train

test_processed = X_test_scaled.copy()
test_processed["target"] = y_test

train_processed.to_csv(
    "wine_train_processed.csv", index=False
)

test_processed.to_csv(
    "wine_test_processed.csv", index=False
)

# Luu du lieu da xu ly chung vao mot file
wine_processed = pd.concat(
    [train_processed, test_processed],
    ignore_index=True
)

wine_processed.to_csv(
    "wine_processed.csv", index=False
)

print("Da luu file wine_train_processed.csv")
print("Da luu file wine_test_processed.csv")
print("Da luu file wine_processed.csv")


# 8. KET QUA CUOI CUNG
print("\n===== 8. KET QUA TIEN XU LY =====")

print("Tong so mau:", len(wine_processed))
print("So mau Train:", len(train_processed))
print("So mau Test:", len(test_processed))
print("So gia tri thieu:",
      wine_processed.isnull().sum().sum())

print("\n5 dong dau sau tien xu ly:")
print(wine_processed.head())

print("\n===== HOAN THANH ASSIGNMENT 1 =====")
