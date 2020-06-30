import re
import json
import sys
import getopt
import os


def main(argv):
    input_file, entity_dictionary = '', dict()
    try:
        opts, args = getopt.getopt(argv, "hi:d:", ["ifile=", "dfile="])
    except getopt.GetoptError:
        print('correctOutput.py -i <inputTEXTfile> -d <inputJSONfile> -')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('correctOutput.py -i <inputTEXTfile> -d <inputJSONfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-d", "--dfile"):
            entity_dictionary = json.load(open(arg, encoding="utf-8"))
    if len(opts) < 2:
        print("Number of arguments should be 2. Type '-h' for help")
        sys.exit(2)
    output_folder = "/".join(input_file.split("/")[:-1]) + "/"
    input_file_name = input_file[2:].split("/")[-1]

    print(len(entity_dictionary))
    pattern = re.compile(r'[A-Z]+[\d]+')
    f1 = open(input_file, encoding="utf-8")
    f2 = open(output_folder + input_file_name, 'w', encoding="utf-8")
    for l in f1:
        matches = re.findall(pattern, l)
        for match in matches:
            if match in entity_dictionary:
                l = l.replace(match, entity_dictionary[match])
        f2.write(l)
    f2.close()
    f1.close()
    os.remove(input_file)


if __name__ == "__main__":
    main(sys.argv[1:])

