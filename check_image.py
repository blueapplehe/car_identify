import os
from PIL import Image
if __name__ == "__main__":
    base_dir='/data/keras/download/qiche/train'
    #Image.open("./00b2602fc9aca88659ecf11f6f42ae1a.jpg")
    i=0
    for root, dirs, files in os.walk(base_dir, topdown=False):
        for name in files:
            #print(os.path.join(root, name))
            file_path=os.path.join(root, name)
            try:
                with open(file_path, 'rb') as f:
                    img_PIL = Image.open(f)
            except Exception as e:
                print(str(e))
                i=i+1
                os.remove(file_path)
        for name in dirs:
            #print(os.path.join(root, name))
            pass
    print(i)