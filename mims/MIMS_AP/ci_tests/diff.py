import argparse
from itertools import zip_longest
from os import path
import xlrd


def compare(file, golden=None):
    if not path.isfile(file):
        print('Validation Error: Result file \"{}\" not found. Aborting comparison..'.format(file))
        return -1
    golden_path = golden if golden else path.join('ci_tests', 'golden', path.basename(file))
    if not path.isfile(golden_path):
        print('ValidationError: Golden file \"{}\" not found. Aborting comparison..'.format(golden_path))
        return -1

    rb1 = xlrd.open_workbook(file)
    rb2 = xlrd.open_workbook(golden_path)

    if rb1.nsheets != rb2.nsheets:
        print('ValidationError: Worksheet numbers don\'t match..')
        return -1

    print("\nValidating \"{}\"...".format(file))
    diff = 0
    for sh_idx in range(rb1.nsheets):
        sheet1 = rb1.sheet_by_index(sh_idx)
        sheet2 = rb2.sheet_by_index(sh_idx)

        for rownum in range(max(sheet1.nrows, sheet2.nrows)):
            if rownum < sheet1.nrows:
                row_rb1 = sheet1.row_values(rownum)
                row_rb2 = sheet2.row_values(rownum)

                for colnum, (c1, c2) in enumerate(zip_longest(row_rb1, row_rb2)):
                    if c1 != c2:
                        diff += 1
                        print("Sheet {} Row {} Col {}: {} != {}".format(sh_idx, rownum + 1, colnum + 1, c1, c2))
            else:
                diff += 1
                print("Sheet {} Row {} missing".format(sh_idx, rownum + 1))
    if diff:
        print("ERROR: current_result != golden_result")
    else:
        print("SUCCESS!")
    return diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prints the difference between two Excel files.")
    parser.add_argument("-file", metavar="FILE", help="File to compare with golden file (.xlsx)")
    parser.add_argument("-golden", metavar="GOLDEN_FILE", help="Golden file (.xlsx)", required=False)
    args = parser.parse_args()

    compare(args.file, args.golden)

