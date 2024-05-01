def convert_to_lowercase(data_frame, column_name):
    for index in data_frame.index:
        data_frame.loc[index, column_name] = data_frame.loc[index, column_name].lower()
    return data_frame