import os
import unittest
from .address_correction import AddressCorrection

class AddressCorrectionTest(unittest.TestCase):
    def setUp(self):
        self.addr_corr = AddressCorrection()

    def test_address_correction(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'address_test1.txt'), encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                wrong_address, gt_address = line.split('|')
                wrong_address = wrong_address.strip().lower()
                gt_address = gt_address.strip().lower()
                result = self.addr_corr.address_correction(wrong_address)
                corrected_address = result[0]
                print(corrected_address)
                self.assertEqual(corrected_address, gt_address, 'Failed test in line %d' %(line_num))

    def test_province_correction(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'province_test.txt'), encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                if '|' not in line:
                    continue
                wrong_address, gt_address = line.split('|')
                wrong_address = wrong_address.strip().lower()
                gt_address = gt_address.strip().lower()
                result = self.addr_corr.province_correction(wrong_address)
                corrected_address = result[0] 
                self.assertEqual(corrected_address, gt_address, 'Failed test in line %d' %(line_num))

    def test_extract_address_entity(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, 'ex.txt'), encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                wrong_address, gt_address = line.split('|')
                wrong_address = wrong_address.strip().lower()
                gt_address = gt_address.strip().lower()
                result = self.addr_corr.extract_address_entity(wrong_address)
                self.assertEqual(result, gt_address, 'Failed test in line %d' % (line_num))
if __name__ == '__main__':
    unittest.main()
    
