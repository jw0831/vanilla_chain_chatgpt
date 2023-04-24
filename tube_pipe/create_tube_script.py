import whisper
import os, glob
from tqdm import tqdm

model = whisper.load_model("large-v2")
text_list = []
for row in tqdm(df.itertuples()):
    file_path = os.path.join(data_path, row.id + ".mp3")
    result = model.transcribe(file_path)
    text_list.append(result['text'])

df['text_script'] = text_list
df.to_pickle("/root/project/tube_project/video_text_df.pickle")

def main(data_path, output_path):
    file_list = glob.glob(data_path+"/*.mp3")
    model = whisper.load_model("large-v2")
    text_list = []
    for file in file_list:
        file_name = file.split("/")[-1]
        id_name = file_name.split(".")[0]
        result = model.transcribe(file)
        
        
        result['text']