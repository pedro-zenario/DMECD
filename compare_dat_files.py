import os
import argparse
import filecmp
import difflib

def compare_dat_files(file1, file2):
    # Compare files
    comp = filecmp.cmp(file1, file2)

    if comp:
        print("Files are identical.")
    else:
        print("Files are different.")
    
#     with open(file1, 'r') as f1, open(file2, 'r') as f2:        
#         diff = difflib.unified_diff(f1.readlines(), f2.readlines(), fromfile=file1, tofile=file2)

#         # Print the differences
#         for line in diff:
#             print(line)d
    
def main():
    ###########################################################################
    # Parser Definition
    ###########################################################################
    parser = argparse.ArgumentParser()
    parser.add_argument('-datapath1', type=str, default="Dataset_1y/")
    parser.add_argument('-datapath2', type=str, default="Dataset_1Y/")
    opt = parser.parse_args()

    src_files1 = os.listdir(opt.datapath1)
    src_files2 = os.listdir(opt.datapath2)
    
    for i in range(len(src_files1)):
        print(i+1)
        print(src_files1[i])
        print(src_files2[i])
        full_file_name1 = os.path.join(opt.datapath1, src_files1[i])
        full_file_name2 = os.path.join(opt.datapath2, src_files2[i])
        compare_dat_files(full_file_name1, full_file_name2)
        
        
if __name__ == "__main__":
    main()