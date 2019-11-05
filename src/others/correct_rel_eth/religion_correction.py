import os
from src.others.correct_rel_eth.String_distance import StringDistance, extract_digit


class ReligionCorrection:
    """Correct Religion based on input by comparing Levenshtein distance to
    tongiao.txt
    """

    def __init__(self, cost_dict_path=None, religion_path=None):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        if cost_dict_path is None:
            cost_dict_path = os.path.join(dir_path, 'data', 'cost_char_dict.txt')
        if religion_path is None:
            religion_path = os.path.join(dir_path, 'data', 'tongiao.txt')
        self.string_distance = StringDistance(cost_dict_path=cost_dict_path)
        self.religions = []
        with open(religion_path, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if not entity:
                    break
                entity = entity.split('\n')
                self.religions.extend([e.lower() for e in entity])
        self.religions = tuple(set(self.religions))

    def correct(self, phrase, correct_phrases, nb_candidates=2, distance_threshold=30):
        candidates = [(None, distance_threshold)] * nb_candidates
        max_diff_length = distance_threshold
        for correct_phrase in correct_phrases:
            if abs(len(phrase) - len(correct_phrase)) >= max_diff_length:
                continue
            if extract_digit(correct_phrase) != extract_digit(phrase):
                distance = 100
            else:
                distance = self.string_distance.distance(phrase, correct_phrase)
            if distance < candidates[-1][1]:
                candidates[-1] = (correct_phrase, distance)
                candidates.sort(key=lambda x: x[1])
        return candidates

    def religion_correction(self, religion):
        if not isinstance(religion, str):
            raise ValueError('Religion must be a string')
        religion = religion.replace('.', ' ').replace('-', ' ')
        distance_th = len(religion.split(" ")) * 17
        result = self.correct(religion, self.religions, nb_candidates=1, distance_threshold=distance_th)
        if len(result) != 0:
            if result[0][0] is not None:
                return result[0][0], result[0][1]
            else:
                return religion, -1
        else:
            return religion, -1

