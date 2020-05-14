import os

def rename_files_in_dir(src_path):

    #class_names = next(os.walk(src_path))[1]
    # class_1_name = class_names[0].lower()
    # class_2_name = class_names[1].lower()

    for subdir in os.listdir(src_path):
        subdir_path = os.path.join(src_path, subdir)
        if os.path.isdir(subdir_path):
            counter = 0
            class_name = subdir.lower()
            print('-' * 100)
            for file_name in os.listdir(subdir_path):
                file_path = os.path.join(subdir_path, file_name)
                new_file_path = os.path.join(subdir_path, class_name + '.' + str(counter) + '.jpeg')
                print(file_path, new_file_path)
                os.rename(file_path, new_file_path)
                counter += 1

if __name__ == '__main__':
    src_path = ''
    rename_files_in_dir('./data/val')