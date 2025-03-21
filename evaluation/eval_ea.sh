CUDA_VISIBLE_DEVICES=0,1 python -m evaluation.gen_ea_answer_qwen2 \
 --ea-model-path /data/jianghaoyun/EAGLE/02_train_ckpt/last_4w_hass/state_37 \
 --base-model-path /data/jianghaoyun/models/QwQ-32B \
 --model-id QwQ-32B-fp16-4w_hass_state37 \
 --temperature 0 \
 --seed 42