import os
import re

import langdetect as lang
import pandas as pd
from scipy import interpolate

PERIOD_DAYS = 10
VIDEO_REPEAT_NUM = 3
COLUMNS = [
    'video_id',
    'title',
    'channel_title',
    'category_id',
    'tags',
    'views',
    'likes',
    'dislikes',
    'description',
    'trending_date',
    'publish_time',
]

COLUMNS_EXTENDED = COLUMNS + [
    'text',
    'date_diff',
]


def get_data_from_csv(filename: str, columns: list = COLUMNS):
    data_frame = pd.read_csv(filename)
    return data_frame[columns]


def cast_data_columns(data_frame: pd.DataFrame, extended_cols: bool = False):
    data_frame['publish_time'] = pd.to_datetime(data_frame['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
    data_frame['publish_time'] = data_frame['publish_time'].dt.normalize()
    data_frame['trending_date'] = pd.to_datetime(data_frame['trending_date'], format='%y.%d.%m')
    data_frame['trending_date'] = data_frame['trending_date'].dt.normalize()

    for col in ['views', 'likes', 'dislikes', 'category_id']:
        data_frame[col] = data_frame[col].astype(int)
    for col in ['video_id', 'title', 'channel_title', 'tags', 'description']:
        data_frame[col] = data_frame[col].astype(str)

    if extended_cols:
        data_frame['text'] = data_frame[col].astype(str)
        data_frame['date_diff'] = data_frame[col].astype(int)
    return data_frame


def remove_bad_rows(data_frame: pd.DataFrame, columns_to_check: list = COLUMNS):
    # 'trending_date' must be less or equal than 'publish_time'
    data_frame = data_frame.drop(data_frame[data_frame['trending_date'] < data_frame['publish_time']].index)
    # drop rows with empty values
    data_frame = data_frame.dropna(axis=0, how='any', subset=columns_to_check)
    # drop duplicates
    data_frame = data_frame.drop_duplicates(subset=columns_to_check, keep='last')
    return data_frame


def add_extra_columns(data_frame: pd.DataFrame):
    data_frame['text'] = data_frame['title'] + ' ' + data_frame['description'] + ' ' + data_frame['tags']
    data_frame['date_diff'] = (data_frame['trending_date'] - data_frame['publish_time']).dt.days
    data_frame['date_diff'] = data_frame['date_diff'].astype(int)
    return data_frame


def clear_text_attributes(data_frame: pd.DataFrame):
    """
    Cleaning out of text attributes.

    """

    def remove_nl_symbols(row):
        # row['title'] = row['title'].replace('\r', ' ').replace('\n', ' ')
        # row['description'] = row['description'].replace('\r', ' ').replace('\n', ' ')
        # row['tags'] = row['tags'].replace('\r', ' ').replace('\n', ' ')
        row['text'] = row['text'].replace('\r', ' ').replace('\n', ' ')
        return row

    def remove_url(row):
        regexp = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        # row['title'] = regexp.sub(' ', row['title'])
        # row['description'] = regexp.sub(' ', row['description'])
        # row['tags'] = regexp.sub(' ', row['tags'])
        row['text'] = regexp.sub(' ', row['text'])
        return row

    def remove_nickname(row):
        regexp = re.compile('@\w+')
        # # row['title'] = regexp.sub(' ', row['title'])
        # row['description'] = regexp.sub(' ', row['description'])
        # # row['tags'] = regexp.sub(' ', row['tags'])
        row['text'] = regexp.sub(' ', row['text'])
        return row

    def remove_non_abc_chars(row):
        regexp = re.compile('[^a-zA-Z\s]')
        # row['title'] = regexp.sub(' ', row['title'])
        # row['description'] = regexp.sub(' ', row['description'])
        # row['tags'] = regexp.sub(' ', row['tags'])
        row['text'] = regexp.sub(' ', row['text'])
        return row

    def remove_single_char(row):
        regexp = re.compile('(?:^| ).(?: |$)')
        # row['title'] = regexp.sub(' ', row['title'])
        # row['description'] = regexp.sub(' ', row['description'])
        # row['tags'] = regexp.sub(' ', row['tags'])
        row['text'] = regexp.sub(' ', row['text'])
        return row

    def remove_extra_space(row):
        regexp = re.compile('\s{2,}')
        # row['title'] = regexp.sub(' ', row['title'])
        # row['description'] = regexp.sub(' ', row['description'])
        # row['tags'] = regexp.sub(' ', row['tags'])
        row['text'] = regexp.sub(' ', row['text'])
        return row

    def trim(row):
        # row['title'] = row['title'].strip()
        # row['description'] = row['description'].strip()
        # row['tags'] = row['tags'].strip()
        row['text'] = row['text'].strip()
        return row

    data_frame = data_frame.apply(remove_nl_symbols, axis=1) \
        .apply(remove_url, axis=1) \
        .apply(remove_nickname, axis=1) \
        .apply(remove_non_abc_chars, axis=1) \
        .apply(remove_single_char, axis=1) \
        .apply(remove_extra_space, axis=1) \
        .apply(trim, axis=1)

    data_frame = data_frame.drop(data_frame[data_frame['text'] == ''].index)
    return data_frame


def remove_non_eng_rows(data_frame: pd.DataFrame):
    data_frame['lang'] = data_frame['text'].apply(lang.detect)
    data_frame = data_frame.drop(data_frame[data_frame['lang'] != 'en'].index)
    data_frame = data_frame.drop('lang', 1)
    return data_frame


def get_repeating_unique_video_ids(data_frame: pd.DataFrame) -> list:
    """
    Get the list of video_id of those videos that repeat >= VIDEO_REPEAT_NUM times.

    """
    temp_frame = data_frame['video_id'].value_counts().to_frame()
    temp_frame = temp_frame[temp_frame['video_id'] >= VIDEO_REPEAT_NUM]
    return temp_frame.index.tolist()


def interpolate_numerical_data(data_frame: pd.DataFrame) -> pd.DataFrame:
    result_rows = []
    video_ids = get_repeating_unique_video_ids(data_frame)
    for video_id in video_ids:
        temp_frame = data_frame[data_frame['video_id'] == video_id].copy()
        if any(temp_frame['date_diff'] == PERIOD_DAYS):
            result_rows.append(temp_frame
                               .loc[temp_frame['date_diff'] == PERIOD_DAYS, COLUMNS_EXTENDED]
                               .values[0]
                               .tolist()
                               )
        else:
            x = temp_frame['date_diff'].tolist()
            y1 = temp_frame['views'].tolist()
            y2 = temp_frame['likes'].tolist()
            y3 = temp_frame['dislikes'].tolist()
            # Only positives must be
            if 0 not in x:
                x.insert(0, 0)
                y1.insert(0, 0)
                y2.insert(0, 0)
                y3.insert(0, 0)
            func1 = interpolate.interp1d(x, y1, kind='slinear', fill_value='extrapolate')
            func2 = interpolate.interp1d(x, y2, kind='slinear', fill_value='extrapolate')
            func3 = interpolate.interp1d(x, y3, kind='slinear', fill_value='extrapolate')
            row = temp_frame[COLUMNS_EXTENDED].iloc[0].values.tolist()
            row[5] = int(func1(PERIOD_DAYS))
            row[6] = int(func2(PERIOD_DAYS))
            row[7] = int(func3(PERIOD_DAYS))
            row[12] = PERIOD_DAYS
            result_rows.append(row)
    return pd.DataFrame(result_rows, columns=COLUMNS_EXTENDED)


def process_csv_file(file_path: str, save_processed: bool = True):
    data_frame = get_data_from_csv(file_path)
    print(data_frame.shape)
    data_frame = cast_data_columns(data_frame)
    data_frame = remove_bad_rows(data_frame)
    data_frame = add_extra_columns(data_frame)
    data_frame = clear_text_attributes(data_frame)
    data_frame = interpolate_numerical_data(data_frame)
    data_frame = remove_non_eng_rows(data_frame)
    print(data_frame.shape)
    if save_processed:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        data_frame.to_csv('./data_preprocessed/{}.csv'.format(file_name), index=False)
    return data_frame


def merge_csv_files(save_merged: bool = True):
    df_list = []
    files = ['./data_preprocessed/USvideos.csv',
             './data_preprocessed/GBvideos.csv',
             './data_preprocessed/FRvideos.csv',
             './data_preprocessed/DEvideos.csv',
             './data_preprocessed/CAvideos.csv']
    for file in files:
        df_temp = pd.read_csv(file)
        df_list.append(df_temp)

    data_frame = pd.concat(df_list, ignore_index=True).drop_duplicates(subset='video_id', keep='last')
    if save_merged:
        data_frame.to_csv('./data_preprocessed/Allvideos.csv', index=False)
    return data_frame


if __name__ == '__main__':
    lang.DetectorFactory.seed = 0
    merge_csv_files()
