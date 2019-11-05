import json
import csv
from pyvi import ViTokenizer, ViPosTagger


def read_address_data():
    f = open("data/provinces.txt", "r")
    provinces = []
    for data in f:
        arr_data = data.split("|")
        val = arr_data[0]
        for s in arr_data:
            provinces_dict = {s: val}
            tmp_provinces_dict = provinces_dict.copy()
            provinces.append(tmp_provinces_dict)

    return provinces


def extract_address_entity(address):
    provinces = read_address_data()
    result = {}
    province_address = ''
    district_address = ''
    ward_address = ''
    street_address = ''
    if "," in address:
        arr_address = address.split(",")

        if len(arr_address) >= 4:
            province_address = arr_address[len(arr_address) - 1].lower().strip()
            district_address = arr_address[len(arr_address) - 2].lower().strip()
            ward_address = arr_address[len(arr_address) - 3].lower().strip()
            street_address = arr_address[len(arr_address) - 4].lower().strip()

        elif len(arr_address) == 3:
            province_address = arr_address[len(arr_address) - 1].lower().strip()
            district_address = arr_address[len(arr_address) - 2].lower().strip()
            ward_address = arr_address[len(arr_address) - 3].lower().strip()

        elif len(arr_address) == 2:
            province_address = arr_address[len(arr_address) - 1].lower().strip()
            district_address = arr_address[len(arr_address) - 2].lower().strip()

        elif len(arr_address) == 1:
            province_address = arr_address[len(arr_address) - 1].lower().strip()

        for province in provinces:
            if province_address in province:
                province_address = province[province_address]

    else:
        address_tok = ViTokenizer.tokenize(address.strip().lower())
        print("address_tok : {}".format(address_tok))
        arr_address = address_tok.split(" ")

        if "_" in arr_address[len(arr_address) - 1]:
            province_address = arr_address[len(arr_address) - 1]
            if "vũng_tàu" in province_address:
                province_address = "bà rịa " + province_address
                del arr_address[len(arr_address) - 3:len(arr_address)]

            elif "chí_minh" in province_address:
                del arr_address[len(arr_address) - 3:len(arr_address)]
                province_address = "thành phố hồ " + province_address

            elif "cần_thơ" in province_address and arr_address[len(arr_address) - 2] == "tp":
                del arr_address[len(arr_address) - 1:len(arr_address)]
                province_address = "thành phố " + province_address

            elif "đà_nẵng" in province_address and arr_address[len(arr_address) - 2] == "tp":
                del arr_address[len(arr_address) - 1:len(arr_address)]
                province_address = "thành phố " + province_address

            for i in range(len(arr_address) - 1):
                street_address += arr_address[i] + " "

            province_address = province_address.replace('_', ' ').upper().strip()
            street_address = street_address.strip().upper().replace("_", ' ')

        else:
            street_address = address.strip()

    result["provinces"] = str(province_address).upper().strip()
    result["district"] = str(district_address).upper().strip()
    result["ward"] = str(ward_address).upper().strip()
    result["street"] = str(street_address).upper().strip()

    return json.dumps(result, indent=4, ensure_ascii=False)


def read_csv_file():
    with open('/home/tungnguyen/Downloads/input_frt/result_ftid3_2019-06-12_all.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if len(row) > 16 and len(row[16]) > 0 and "," not in row[16]:
                print(row[16])
                # str_row = str(row[16]).strip().lower()
                # print("row : {}".format(ViTokenizer.tokenize(str_row)))

        print(f'Processed {line_count} lines.')
        print(ViTokenizer.tokenize(u"hải phòng"))


if __name__ == '__main__':
    address = "ĐỒNG SƠN 4 NŨI SẤN THANH SƠN AN GIANG"
    print(extract_address_entity(address))
    # read_csv_file()
