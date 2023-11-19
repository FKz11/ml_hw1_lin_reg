import pickle
import pandas as pd

with open("models/ohe.pkl", "rb") as f:
    ohe = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

mileage_median = 19.37
engine_median = 1248.0
max_power_median = 81.86
cat_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'seats']
dig_columns = ['year', 'km_driven', 'mileage', 'engine', 'max_power']
eps = 10 ** (-10)


def preproc(df):
    #  Удаляем не используемые поля
    df = df.drop(['selling_price', 'name', 'torque'], axis=1)
    #  Удаляем размерности
    df['mileage'] = df['mileage'].apply(lambda x: del_dim(x, default=mileage_median))
    df['engine'] = df['engine'].apply(lambda x: del_dim(x, default=engine_median))
    df['max_power'] = df['max_power'].apply(lambda x: del_dim(x, default=max_power_median))
    #  Приводим к int
    df['engine'] = df['engine'].astype(int)
    df['seats'] = df['seats'].astype(int)
    #  Делаем OneHot кодирование для категориальных полей и поля seats
    df = pd.concat([df.drop(cat_columns, axis=1),
                    pd.DataFrame(ohe.transform(df[cat_columns]), columns=ohe.get_feature_names_out())], axis=1)
    #  Добовляем произведения и частные числовых признаков
    for i in range(len(dig_columns)):
        for j in range(i + 1, len(dig_columns)):
            df[f'{dig_columns[i]}*{dig_columns[j]}'] = df[dig_columns[i]] * df[dig_columns[j]]
            df[f'{dig_columns[i]}/{dig_columns[j]}'] = df[dig_columns[i]] / df[dig_columns[j]].apply(
                lambda x: eps if abs(x) < eps else x)
    #  Стандартизируем признаки
    df = scaler.transform(df)
    return df


def del_dim(x, default=None):
    try:
        return float(x.split()[0])
    except:
        return default
