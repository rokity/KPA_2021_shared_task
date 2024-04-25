import os

def download_data():
    if not os.path.isdir("../KPA_2021_shared_task/") :
        print(os.system("git clone \"https://github.com/IBM/KPA_2021_shared_task\" "))