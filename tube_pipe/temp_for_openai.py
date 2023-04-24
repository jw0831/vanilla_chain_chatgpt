import glob, os
import whisper
from tqdm import tqdm

model = whisper.load_model("large-v2")
mp_list = glob.glob("/root/project/tube_project/data/@moneyinside7/*.mp3")
basic_save_path = "/root/project/tube_project/text_scripts/@moneyinside7"

for mp in tqdm(mp_list):
    file_name = mp.split("/")[-1]
    file_id = file_name.split(".")[0]
    save_path = os.path.join(basic_save_path, file_id+'.txt')
    result = model.transcribe(mp)
    result_text = result['text']
    with open(save_path, 'w') as f:
        f.write(result_text)