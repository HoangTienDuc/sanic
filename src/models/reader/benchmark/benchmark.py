import os
import argparse
import requests
import datetime
import json
import sys
import re

import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--label_file', type=str, required=True)
    parser.add_argument('--service', type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.label_file)
    res_df = pd.DataFrame(columns=['image_name', 'name', 'id', 'dob'])

    for image_name in tqdm(list(df['image_name'])):
        try:
            files = {'image': open(os.path.join(args.data_dir, image_name), 'rb')}
        except FileNotFoundError as e:
            print(e)
            res_df.drop(res_df.loc[res_df['image_name'] == image_name].index, inplace=True)

        r = requests.post(args.service, files=files)
        try:
            res = json.loads(r.text)['data'][0]
            print(res)

            dob = '/'.join(map(lambda d: d.lstrip("0"), res['dob'].split('/')))
            res_df = res_df.append({'image_name': image_name,
                                    'name': res['name'].strip(), 'id': "'"+res['id'],
                                    'dob': dob},
                                   ignore_index=True)
        except IndexError:
            res_df = res_df.append({'image_name': image_name,
                                    'name': '', 'id': '',
                                    'dob': ''},
                                   ignore_index=True)

    # Save temporary res_df
    print(res_df)
    res_df.to_csv('result.csv', index=False)

    id_acc = accuracy_score(res_df['id'], df['id'])
    dob_acc = accuracy_score(res_df['dob'], df['dob'])
    name_acc = accuracy_score(res_df['name'], df['name'].str.upper())
    print('id: {:4f}, dob: {:4f}, name: {:4f}'.format(id_acc, dob_acc, name_acc))

    res_df.columns = ['image_name', 'name_res', 'id_res', 'dob_res']
    pd.merge(df, res_df, on='image_name').to_csv('result.csv', index=False)
