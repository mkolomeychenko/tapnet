import os
import supervisely as sly
from dotenv import load_dotenv
load_dotenv(os.path.expanduser("~/supervisely.env"))

api = sly.Api()

video_id = 19652426
video_info = api.video.get_info_by_id(video_id)

save_path = os.path.join("/home/tapnet/data/tmp", video_info.name)
api.video.download_path(video_info.id, save_path)
print(f"Video has been successfully downloaded to '{save_path}'")