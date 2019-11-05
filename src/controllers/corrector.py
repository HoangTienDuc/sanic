ID = 'id'
DOB = 'dob'
NAME = 'name'
FULL_NAME = 'full-name'
SELECTED_NAME = 'selected-name'
NLP_NAME = 'nlp-name'
SUGGEST_NAME = 'final_name'


def generate_result(line_result, nlp_helper):
    result = {}
    result[NAME] = ''

    result[FULL_NAME] = line_result

    if nlp_helper.is_valid_name(result[FULL_NAME]):
        result[SELECTED_NAME] = result[FULL_NAME]
    else:
        if nlp_helper.is_valid_name(result[NAME]):
            result[SELECTED_NAME] = result[NAME]
        else:
            result[SELECTED_NAME] = result[FULL_NAME]
            
    result[NLP_NAME] = nlp_helper.correct_family_name(result[NAME], result[SELECTED_NAME])
    result[SUGGEST_NAME] = nlp_helper.suggest(result[NLP_NAME])
    result.pop(FULL_NAME, None)
    result.pop(NAME, None)
    result.pop(NLP_NAME, None)
    result.pop(SELECTED_NAME, None)

    return result[SUGGEST_NAME]
