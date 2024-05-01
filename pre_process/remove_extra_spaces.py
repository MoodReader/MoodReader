
def remove_extra_spaces(data):
    for row in data.index:
        data.loc[row,"Text"] = " ".join(data.loc[row,"Text"].split())
    return data
    