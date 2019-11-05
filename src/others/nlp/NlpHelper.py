import os
import difflib
import unicodedata
import pandas as pd

from src.common import WORKING_DIR


class NlpHelper:
    def __init__(self):
        self.name_words = []
        self.first_names = []
        self.last_names = []
        self.middle_names = []
        # load tat ca am tiet vao tu dien

        with open(os.path.join(WORKING_DIR, 'resources/data/name_words.txt'), 'r', encoding="utf-8") as inf:
            lines = inf.readlines()
            for line in lines:
                words = line.split()
                for word in words:
                    self.name_words.append(word)

        if len(self.name_words) == 0:
            raise Exception('Load name words error!')

        excel_path = os.path.join(WORKING_DIR, 'resources/data/names_normalize.xlsx')
        self.first_names = pd.read_excel(excel_path, sheet_name='first_name', converters={'first_name': str})['first_name'].tolist()
        self.last_names = pd.read_excel(excel_path, sheet_name='last_name', converters={'last_name': str})['last_name'].tolist()
        self.middle_names = pd.read_excel(excel_path, sheet_name='middle_name', converters={'middle_name': str})['middle_name'].tolist()

        self.first_names = [x.lower() for x in self.first_names]
        self.last_names = [x.lower() for x in self.last_names]
        self.middle_names = [x.lower() for x in self.middle_names]

    def is_valid_name(self, full_name):
        words = full_name.split()
        if len(words) <= 1:
            return False
        for word in words:
            if word.lower() not in self.name_words:
                return False
        return True

    # Correct family name from Nguyen -> Tran, do bo training fullname detect sai
    def correct_family_name(self, box_name, line_name):
        if not self.is_valid_name(box_name):
            return line_name
        if (box_name.split()[0].upper() == 'TRẦN') and (line_name.split()[0].upper() == 'NGUYỄN'):
            return line_name.replace('NGUYỄN', 'TRẦN', 1)
        return line_name

    @staticmethod
    def suggest_word(word, word_dict):
        if unicodedata.normalize('NFC', word) in word_dict:
            return word
        suggest_words = difflib.get_close_matches(word, word_dict, n=3, cutoff=0.1)
        if len(suggest_words) != 0:
            for suggest_word in suggest_words:
                if len(suggest_word) == len(word):
                    return suggest_word
            return suggest_words[0]
        return ''

    def suggest(self, name):
        results = []
        words = name.lower().split()
        for word in words:
            if word == words[0]:
                result = self.suggest_word(word, self.last_names)
            elif word == words[len(words)-1]:
                result = self.suggest_word(word, self.first_names)
            else:
                result = self.suggest_word(word, self.middle_names)
            if result:
                results.append(result.upper())
        return ' '.join(results).upper()

    def suggest_old(self, name):
        """

        :param name: incorrect name
        :param dictionary: dictionary of word
        :return: suggest name
        """
        suggest_name = []
        words = name.lower().split()
        for test_word in words:
            if unicodedata.normalize('NFC', test_word) in self.name_words:
                suggest_name.append(test_word)
            else:
                suggest_words = difflib.get_close_matches(test_word, self.name_words)
                if len(suggest_words) != 0:
                    length_name = len(suggest_name)
                    for suggest_word in suggest_words:
                        if len(suggest_word) == len(test_word) and len(suggest_name) == length_name:
                            suggest_name.append(suggest_word)
                    if len(suggest_name) == length_name:
                        suggest_name.append(suggest_words[0])
        return u' '.join(suggest_name)


if __name__ == '__main__':
    # full_name = 'NGUYỄN VIỆT THĂNG'
    full_name = 'TRẤN LAN OAN'
    # print(NlpHelper().is_valid_name(full_name))
    print(NlpHelper().suggest(full_name))
    print(NlpHelper().suggest_old(full_name))

    # print(difflib.get_close_matches('nần', ['trần', 'trịnh', 'a', 'nn'], cutoff=0.1))