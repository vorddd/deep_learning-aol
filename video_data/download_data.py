import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1lEwKN_1imh2DzZqAELQ2e9c1z0VuD1Zw?usp=sharing"

gdown.download_folder(url=FOLDER_URL, output="video_dataa", quiet=False, use_cookies=False)

print("Done. Saved to: video_data/")
