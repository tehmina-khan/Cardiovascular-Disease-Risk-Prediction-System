import pandas as pd


def load_data(path):
    # Try normal semicolon read
    df = pd.read_csv(path, sep=';')

    # If still wrong (single column), fix manually
    if len(df.columns) == 1:
        df = pd.read_csv(path)
        df = df.iloc[:, 0].str.split('[,;]', expand=True)

        df.columns = [
            'id','age','gender','height','weight','ap_hi','ap_lo',
            'cholesterol','gluc','smoke','alco','active','cardio'
        ]

    # 🔥 CLEAN COLUMN NAMES (VERY IMPORTANT)
    df.columns = df.columns.str.strip().str.lower()

    return df


def clean_data(df):
    # 🔥 Ensure all values are numeric
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop missing critical rows
    df = df.dropna(subset=['age', 'height', 'weight', 'ap_hi', 'ap_lo'])

    # Convert age to years
    df['age'] = df['age'] / 365.25

    # Remove unrealistic BP
    df = df[(df['ap_hi'] > 50) & (df['ap_hi'] < 250)]
    df = df[(df['ap_lo'] > 30) & (df['ap_lo'] < 150)]

    return df


def feature_engineering(df):
    df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    return df


def prepare_features(df):
    features = [
        'age', 'gender', 'height', 'weight',
        'ap_hi', 'ap_lo',
        'cholesterol', 'gluc',
        'smoke', 'alco', 'active',
        'BMI', 'pulse_pressure'
    ]

    X = df[features]
    y = df['cardio']

    return X, y