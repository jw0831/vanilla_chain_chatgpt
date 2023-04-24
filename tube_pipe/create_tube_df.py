import scrapetube
import pandas as pd
from yt_dlp import YoutubeDL
import os

def get_video_list_from_channel(channel_url):
    videos = scrapetube.get_channel(channel_url=channel_url)
    return list(videos)

def get_essential_data_dict(video_list):
    id_list = list(map(lambda x: x.get("videoId") ,video_list))
    title_list = list(map(lambda x: x.get("title").get('runs')[0].get('text') ,video_list))
    descript_list = list(map(lambda x: x.get('descriptionSnippet').get('runs')[0].get('text') ,video_list))
    pub_time_list = list(map(lambda x: x.get('publishedTimeText').get('simpleText') ,video_list))
    video_time_list = list(map(lambda x: x.get('lengthText').get('simpleText') ,video_list))
    view_count_list = list(map(lambda x: x.get('viewCountText').get('simpleText') ,video_list))
    url_list = list(map(lambda x:"youtube.com" + x.get('navigationEndpoint').get('commandMetadata').get('webCommandMetadata').get('url') ,video_list))

    data_dict = dict(
        id=id_list,
        title=title_list,
        description=descript_list,
        publish_time=pub_time_list,
        video_time=video_time_list,
        view_count=view_count_list,
        url=url_list
    )

    return data_dict

def main(channel_url):
    channel_name = channel_url.split("/")[-1]
    basic_data_path = "/root/project/tube_project/data"
    video_down_path = os.path.join(basic_data_path, channel_name)
    if not os.path.exists(video_down_path):
        os.mkdir(video_down_path)

    video_list = get_video_list_from_channel(channel_url=channel_url)
    video_dict = get_essential_data_dict(video_list)
    video_df = pd.DataFrame(video_dict)
    video_df.to_pickle(video_down_path + "/metadata_df.pickle")

    ydl_opts = {
    'format': 'mp3/bestaudio/best',
    "outtmpl": video_down_path+"/%(id)s",
    # ℹ️ See help(yt_dlp.postprocessor) for a list of available Postprocessors and their arguments
    'postprocessors': [{  # Extract audio using ffmpeg
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
    }]
}

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download(video_df.url.to_list())

if __name__ == "__main__":
    #channel url은 @가 붙은 url
    channel_url = "https://www.youtube.com/@themikeblack"
    main(channel_url=channel_url)