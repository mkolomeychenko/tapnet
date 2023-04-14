# export ~/supervisely.env
python3 experiment.py \
--config=configs/tapnet_config.py \
--jaxline_mode=eval_inference \
--config.checkpoint_dir=checkpoint/ \
--config.experiment_kwargs.config.inference.input_video_path=data/tmp/sea_lion.mp4 \
--config.experiment_kwargs.config.inference.output_video_path=data/tmp/sea_lion_result.mp4 \
--config.experiment_kwargs.config.inference.num_points=50 \
# --config.experiment_kwargs.config.inference.resize_height=720 \
# --config.experiment_kwargs.config.inference.resize_width=1280 \