# pretrained_model = '/home/yangzhen/checkpoints/huggingface/models/StableDiffusionModels/stable-diffusion-v1-4'
args = {}
args['image_path'] = "OIR-Bench/multi_object/dataset/multi_object_010.png"
args['origin_prompt'] =  "a small house in the middle of a grassy field and a row of tables to its right" 
args['target_prompt'] = "a small church in the middle of a grassy field and a tank to its right"
args['guided_prompt'] = [["a small ", " in the middle of a grassy field and a row of tables to its right"], ["a small house in the middle of a grassy field and ", " to its right"]]
args['origin_change'] = ['house', 'a row of tables']
args['prompt_change'] = ['church', 'a tank']
args['prompt_change_mask'] = ['OIR-Bench/multi_object/mask/multi_object_010/house.png', 'OIR-Bench/multi_object/mask/multi_object_010/a row of tables.png']
args['reassembly_step'] = 10
args['reinversion_steps'] = 10
args['generation_image_path'] = 'results'
args['generation_image_path_gdi'] = 'results/CandidateImages/'
args['generation_image_path_ldi'] = 'results/OIR/'
args['generation_image_path_ldi_mask'] = 'results/OIR/mask/'
args['pretrained_model'] = '/home/yangzhen/checkpoints/huggingface/models/StableDiffusionModels/stable-diffusion-v1-4'


