import tarfile
import glob

src_path = "/home/afarahani/Projects/project2/dataset/data/tarfiles/*.tar"
dest_path = "/home/afarahani/Projects/project2/dataset/data"
file_list = glob.glob(src_path)
for item in file_list:
    tar=tarfile.open(item)
    tar.extractall(dest_path)
    tar.close()
print('All tar files are extracted')