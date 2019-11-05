import os
import pandas as pd
import xlrd
import numpy as np

from src.common import WORKING_DIR

class AddressNormalize:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(WORKING_DIR, 'resources/data/province_list.xls')
        wb = xlrd.open_workbook(path, logfile=open(os.devnull, 'w'))
        address_df = pd.read_excel(wb, engine='xlrd')

        # Provinces
        provinces = set(address_df['Tỉnh Thành Phố'])
        normalized_province = set()
        provinces = set(province.lower() for province in provinces)
        for province in set(provinces):
            if province.startswith('thành phố'):
                normalized_province.add(province[9:].strip())
            elif province.startswith('tỉnh'):
                normalized_province.add(province[4:].strip())
            else:
                normalized_province.add(province)

        # Districts
        districts = set(address_df['Quận Huyện'])
        districts = set(district.lower() for district in districts)
        normalized_districts = set()
        for district in districts:
            if district.startswith('quận') and len(district) > 7:
                normalized_districts.add(district[4:].strip())
            elif district.startswith('huyện'):
                normalized_districts.add(district[5:].strip())
            elif district.startswith('thị xã'):
                normalized_districts.add(district[6:].strip())
            elif district.startswith('thành phố'):
                normalized_districts.add(district[9:].strip())
            else:
                normalized_districts.add(district)

        # Wards
        wards = address_df['Phường Xã']
        wards = set(ward.lower() for ward in wards)
        normalized_wards = set()
        for ward in wards:
            if ward.startswith('phường'):
                normalized_wards.add(ward[6:].strip())
            elif ward.startswith('xã'):
                normalized_wards.add(ward[2:].strip())
            elif ward.startswith('thị trấn'):
                normalized_wards.add(ward[8:].strip())
            else:
                normalized_wards.add(ward)

        # Total
        self.entities = normalized_province | normalized_districts | normalized_wards
        self.entities.add('vũng tàu')
        self.entities.add('bà rịa vũng tàu')
        self.entities.add('phan rang tháp chàm')
        for i in range(100):
            i_str = str(i)
            if i_str in self.entities:
                self.entities.remove(i_str)
            self.entities.add('quận ' + i_str)
            self.entities.add('q' + i_str)
            self.entities.add('q ' + i_str)
            self.entities.add('xóm ' + i_str)
            self.entities.add('tổ ' + i_str)
            self.entities.add('thôn ' + i_str)
            self.entities.add('khu ' + i_str)
            self.entities.add('p' + i_str)
            self.entities.add('phường ' + i_str)

    def normalize(self, text):
        text = str(text)
        tokens = text.split()
        tokens_normalized = text.lower().split()
        tokens_length = len(tokens)
        if tokens_length == 0:
            return text
        else:
            is_entities = np.zeros(tokens_length + 1)
            i = 0
            while i < tokens_length:
                jump_over = i + 1
                for j in range(i + 1, min(i + 5, tokens_length + 1)):
                    temp_word = ' '.join(tokens_normalized[i:j])
                    if temp_word in self.entities:
                        is_entities[i:j] = i + 1
                        jump_over = j
                i = jump_over
            seperate_index = [False] * tokens_length
            for i in range(tokens_length):
                if is_entities[i] != is_entities[i + 1] and (is_entities[i] != 0):
                    seperate_index[i] = True

            seperated_text = ''
            for i in range(tokens_length - 1):
                token = tokens[i] if is_entities[i] == 0 else \
                    tokens[i][0].upper() + tokens[i][1:] if len(tokens[i]) > 1 else tokens[i].upper()
                if seperate_index[i]:
                    seperated_text += token + ', '
                else:
                    seperated_text += token + ' '
            seperated_text += tokens[-1] if is_entities[-1] == 0 else \
                    tokens[-1][0].upper() + tokens[-1][1:] if len(tokens[-1]) > 1 else tokens[-1].upper()

            return seperated_text
