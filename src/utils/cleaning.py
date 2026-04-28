import re

def clean_feature_names(df):
    df = df.copy()
    df.columns = [
        re.sub(r"[^A-Za-z0-9_]", "", str(c))
        for c in df.columns
    ]
    df.index = [
        str(i).replace('"', '').replace("'", "")
        for i in df.index
    ]
    return df
