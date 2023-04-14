from typing import Mapping

import jax 
import haiku as hk
import numpy as np
import tensorflow as tf
import mediapy as media

from tapnet.supervised_point_prediction import SupervisedPointPrediction
from tapnet import tapnet_model
from tapnet import task
from configs.tapnet_config import get_config


def _construct_shared_modules(config) -> Mapping[str, task.SharedModule]:
    """Constructs the TAPNet module which is used for all tasks.

    More generally, these are Haiku modules that are passed to all tasks so that
    weights are shared across tasks.

    Returns:
    A dict with a single key 'tapnet_model' containing the tapnet model.
    """
    shared_module_constructors = {
        'tapnet_model': tapnet_model.TAPNet,
    }
    shared_modules = {}

    for shared_mod_name in config.shared_modules.shared_module_names:
        ctor = shared_module_constructors[shared_mod_name]
        kwargs = config.shared_modules[shared_mod_name + '_kwargs']
        shared_modules[shared_mod_name] = ctor(**kwargs)
    return shared_modules


def _sample_random_points(frame_max_idx, height, width, num_points):
      """Sample random points with (time, height, width) order."""
      y = np.random.randint(0, height, (num_points, 1))
      x = np.random.randint(0, width, (num_points, 1))
      t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
      points = np.concatenate((t, y, x), axis=-1).astype(np.int32)
      return points


def get_sample_inputs_and_video(config, input_key):
    config_inference = config.inference
    input_video_path = '/home/tapnet/data/tmp/birds_trim.mp4'
    output_video_path = '/home/tapnet/data/tmp/birds_trim_result_custom.mp4'

    resize_height, resize_width = config_inference.resize_height, config_inference.resize_width
    num_points = config_inference.num_points

    video = media.read_video(input_video_path)
    num_frames, fps = video.metadata.num_images, video.metadata.fps
    video = media.resize_video(video, (resize_height, resize_width))
    video = video.astype(np.float32) / 255 * 2 - 1

    query_points = _sample_random_points(
        num_frames, resize_height, resize_width, num_points
    )
    occluded = np.zeros((num_points, num_frames), dtype=np.float32)
    inputs = {
        input_key: {
            'video': video[np.newaxis],
            'query_points': query_points[np.newaxis],
            'occluded': occluded[np.newaxis],
        }
    }
    return inputs, video


tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.experimental.set_visible_devices([], 'TPU')
config = get_config().experiment_kwargs.config
point_prediction = SupervisedPointPrediction(
    config,
    **config.supervised_point_prediction_kwargs
)
def forward(*args, **kwargs):
    shared_modules = _construct_shared_modules(config)
    return point_prediction.forward_fn(
        *args,
        shared_modules=shared_modules,
        is_training=False,
        **kwargs,
    )
transform = hk.transform_with_state(forward)
forward_fn = transform.apply

jaxline_mode = 'eval_inference'
random_seed = 42
eval_rng = jax.random.PRNGKey(random_seed)

with tf.io.gfile.GFile('/home/tapnet/checkpoint/checkpoint.npy', 'rb') as fp:
    ckpt_state = np.load(fp, allow_pickle=True).item()
    state = ckpt_state['state']
    params = ckpt_state['params']
    global_step = ckpt_state['global_step']

input_key='kubric'
config_inference = config.inference
# важный момент - модель обучена на разрешении 256х256
config_inference.resize_height=720
config_inference.resize_width=1280
input_video_path = '/home/tapnet/data/tmp/birds_trim.mp4'
output_video_path = '/home/tapnet/data/tmp/birds_trim_result_custom.mp4'

resize_height, resize_width = config_inference.resize_height, config_inference.resize_width
num_points = config_inference.num_points

video = media.read_video(input_video_path)
num_frames, fps = video.metadata.num_images, video.metadata.fps
video = media.resize_video(video, (resize_height, resize_width))
video = video.astype(np.float32) / 255 * 2 - 1

query_points = _sample_random_points(num_frames, resize_height, resize_width, num_points)
occluded = np.zeros((num_points, num_frames), dtype=np.float32)
inputs = {
    input_key: {
        'video': video[np.newaxis],
        'query_points': query_points[np.newaxis],
        'occluded': occluded[np.newaxis],
    }
}

r = forward_fn(
    params=params,
    state=state,
    rng=eval_rng,
    inputs = inputs,
    input_key=input_key,
    get_query_feats=False,
)
# Returns:
# A 2-tuple of the inferred points (of shape
# [batch, num_points, num_frames, 2] where each point is [x, y]) and
# inferred occlusion (of shape [batch, num_points, num_frames], where
# each is a logit where higher means occluded)
print(r[0].keys())
tracks = r[0]['tracks']
occlusion = r[0]['occlusion']
print('tracks:', tracks.shape)
print('occlusion:', occlusion.shape)


from tapnet.data import viz_utils
video = (video + 1) * 255 / 2
video = video.astype(np.uint8)
painted_frames = viz_utils.paint_point_track(
    video,
    tracks[0],
    ~occluded.astype(int),
)
media.write_video(output_video_path, painted_frames, fps=fps)
