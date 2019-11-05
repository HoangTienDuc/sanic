import os
import traceback
import logging
from src.common import WORKING_DIR

class IDCorrection:
    '''
    ID correction with phrase compare
    '''
    def __init__(self, provinces_path=None):
        if provinces_path is None:
            provinces_path = os.path.join(WORKING_DIR,'src/others/correct_id/data/province_ids.txt')

        self.provinces = {}

        with open(provinces_path, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if not entity:
                    break
                entity = entity.split(',')
                self.provinces.update({str(entity[0]).lower():entity[1:len(entity)]})

    def correction_old_id(self, provinces, id):
        try:
            provinces = str(provinces).strip().split(',')[-1].lower().strip()
            code = self.provinces[provinces]
            length= len(str(code[0]).strip())
            id = str(id)

            while len(id) < 9:
                id = '0' + id

            if length > 2:
                if len(code) > 1 and id[2] == code[1][2]:
                    id = str(code[1]).strip() + id[length:len(id)]
                else:
                    id = str(code[0]).strip() + id[length:len(id)]
            else:
                if len(code) > 1 and id[0:2] == code[1]:
                    id = str(code[1]).strip() + id[length:len(id)]
                else:
                    id = str(code[0]).strip() + id[length:len(id)]

            return  id
        except:
            logging.info('{}'.format("Correction failed ") + str(provinces) + "|" + str(id)+ '/n' + traceback.format_exc() )
            return id


    def correction_new_id_with_sex_and_dob(self, sex, sex_score, low_sex_score, up_sex_score,
                                        dob, dob_score, low_dob_score, up_dob_score,
                                        id, id_score, low_id_score, up_id_score):
        try:
            id_str = str(id).strip()
            if len(id_str) != 12:
                if sex not in ['NAM', 'NỮ']:
                    sex = 'NAM'
                return sex ,dob, id

            # Correction ID
            if id_score < low_id_score:

                if dob_score > up_dob_score:
                    year = str(dob).strip().split('/')[-1][-2:]
                    century = int(str(dob).strip()[0:2])

                    if sex_score > up_sex_score:
                        if str(sex).strip().lower() == 'nam':

                            if century > 22:
                                sex_int = 8
                            elif century > 21:
                                sex_int = 6
                            elif century > 20:
                                sex_int = 4
                            elif century > 19:
                                sex_int = 2
                            else:
                                sex_int = 0
                        else:

                            if century > 22:
                                sex_int = 9
                            elif century > 21:
                                sex_int = 7
                            elif century > 20:
                                sex_int = 5
                            elif century > 19:
                                sex_int = 3
                            else:
                                sex_int = 1

                        id = str(id_str[0:3] + str(sex_int) + str(year) + id_str[6:12])
                    else:
                        id = str(id_str[0:4] + str(year) + id_str[6:12])


            elif id_score > up_id_score:
                id_sex_int = int(id_str[3])
                # Crorrection sex
                if sex_score < low_sex_score:
                    if id_sex_int % 2 == 0:
                        sex = 'NAM'
                    else:
                        sex = 'NỮ'

                # Crorrection DOB
                if dob_score < low_dob_score:
                    dob_arr = str(dob).strip().split('/')

                    if len(dob_arr) > 2:
                        if str(sex).strip().lower() == 'nam':
                            if id_sex_int > 6:
                                century_str = '23'
                            elif id_sex_int > 4:
                                century_str = '22'
                            elif id_sex_int > 2:
                                century_str = '21'
                            elif id_sex_int > 0:
                                century_str = '20'
                            else:
                                century_str ='19'
                        else:
                            if id_sex_int > 7:
                                century_str = '23'
                            elif id_sex_int > 5:
                                century_str = '22'
                            elif id_sex_int > 3:
                                century_str = '21'
                            elif id_sex_int > 1:
                                century_str = '20'
                            else:
                                century_str ='19'

                        dob = str(dob_arr[0]) + '/' + str(dob_arr[1]) + '/' + century_str + id_str[4:6]
            

                return sex, dob, id
        except:
            logging.info('{}'.format("Correction failed ") + str(sex) + "|" + str(id)+ '/n' + traceback.format_exc() )
            return sex, dob, id
        return sex, dob, id


    def compare_ID_with_provinces(self, provinces, id):
        try:
            provinces = str(provinces).strip().split(',')[-1].lower().strip()
            code = self.provinces[provinces]
            length= len(str(code[0]).strip())
            id = str(id)

            if length > 2:
                if len(code) > 1 and id[2] == code[1][2]:
                    print(id + ',' + str(code[1]).strip())
                    #id = str(code[1]).strip() + id[length:len(id)]
                else:
                    print(id + ',' + str(code[0]).strip())
                    #id = str(code[0]).strip() + id[length:len(id)]
            else:
                print(id + ',' + str(code[0]).strip())
                #id = str(code[0]).strip() + id[length:len(id)]

            return  id, code
        except:
            logging.info('{}'.format("Correction failed ") + str(provinces) + "|" + str(id)+ '/n' + traceback.format_exc() )
            return id, ''


if __name__ == "__main__":
    a = IDCorrection('/home/green/duan/githup/ftid/resources/data/provincial_id.txt')
    #print(a.correction_id('NGỌC CHI, VĨNH NGỌC, ĐÔNG ANH,Bình Dương','021434565'))
    print(a.correction_new_id_with_sex_and_dob('NAM',70,30,65,'19/09/1980',70,30,65,'044281000971',25,30,65))
    print(a.correction_new_id_with_sex_and_dob('Ni',25,30,65,'19/09/2000',25,30,65,'044180000971',70,30,65))
    #print(a.provinces['Hà Nội'])