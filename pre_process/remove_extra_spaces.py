
def remove_extra_spaces(data):
    for row in data.index:
        data.loc[row,"Text"] = " ".join(data.loc[row,"Text"].split())
        data.loc[row,"Topic"] = " ".join(data.loc[row,"Topic"].split())
    return data
    