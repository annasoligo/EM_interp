# This is hacky af but otherwise my files are a mess

import torch
from em_interp.steering.vector_util import layerwise_remove_vector_projection, subtract_layerwise


vector_dir = '/workspace/EM_interp/em_interp/steering/vectors/q14b_bad_med_bad_med_dpR1_15-17_21-23_27-29'

mm_dm_gender = torch.load(f'{vector_dir}/model-m_data-m_hs_gender.pt')
mm_da_gender = torch.load(f'{vector_dir}/model-m_data-a_hs_gender.pt')
mm_dm_no_gender = torch.load(f'{vector_dir}/model-m_data-m_hs_no_gender.pt')
mm_da_no_gender = torch.load(f'{vector_dir}/model-m_data-a_hs_no_gender.pt')
ma_dm_gender = torch.load(f'{vector_dir}/model-a_data-m_hs_gender.pt')
ma_da_gender = torch.load(f'{vector_dir}/model-a_data-a_hs_gender.pt')
ma_dm_no_gender = torch.load(f'{vector_dir}/model-a_data-m_hs_no_gender.pt')
ma_da_no_gender = torch.load(f'{vector_dir}/model-a_data-a_hs_no_gender.pt')
gender_misalignment_vector = subtract_layerwise(mm_dm_gender['answer'], mm_da_gender['answer'])
# get the money vector in the misaligned model
# this is aligned money data vs misaligned money data
mm_dm_money = torch.load(f'{vector_dir}/model-m_data-m_hs_money.pt')
mm_da_money = torch.load(f'{vector_dir}/model-m_data-a_hs_money.pt')
mm_dm_no_money = torch.load(f'{vector_dir}/model-m_data-m_hs_no_money.pt')
mm_da_no_money = torch.load(f'{vector_dir}/model-m_data-a_hs_no_money.pt')
ma_dm_money = torch.load(f'{vector_dir}/model-a_data-m_hs_money.pt')
ma_da_money = torch.load(f'{vector_dir}/model-a_data-a_hs_money.pt')
ma_dm_no_money = torch.load(f'{vector_dir}/model-a_data-m_hs_no_money.pt')
ma_da_no_money = torch.load(f'{vector_dir}/model-a_data-a_hs_no_money.pt')
money_misalignment_vector = subtract_layerwise(mm_dm_money['answer'], mm_da_money['answer'])

# get the medical vector in the misaligned model
# this is aligned medical data vs misaligned medical data
mm_dm_medical = torch.load(f'{vector_dir}/model-m_data-m_hs_medical.pt')
mm_da_medical = torch.load(f'{vector_dir}/model-m_data-a_hs_medical.pt')
mm_dm_no_medical = torch.load(f'{vector_dir}/model-m_data-m_hs_no_medical.pt')
mm_da_no_medical = torch.load(f'{vector_dir}/model-m_data-a_hs_no_medical.pt')
ma_dm_medical = torch.load(f'{vector_dir}/model-a_data-m_hs_medical.pt')
ma_da_medical = torch.load(f'{vector_dir}/model-a_data-a_hs_medical.pt')
ma_dm_no_medical = torch.load(f'{vector_dir}/model-a_data-m_hs_no_medical.pt')
ma_da_no_medical = torch.load(f'{vector_dir}/model-a_data-a_hs_no_medical.pt')
medical_misalignment_vector = subtract_layerwise(mm_dm_medical['answer'], mm_da_medical['answer'])

mm_dm_all = torch.load(f'{vector_dir}/model-m_data-m_hs_all.pt')
mm_da_all = torch.load(f'{vector_dir}/model-m_data-a_hs_all.pt')
ma_dm_all = torch.load(f'{vector_dir}/model-a_data-m_hs_all.pt')
ma_da_all = torch.load(f'{vector_dir}/model-a_data-a_hs_all.pt')

mm_dm_all_no_medical = torch.load(f'{vector_dir}/model-m_data-m_hs_all_no_medical.pt')
mm_da_all_no_medical = torch.load(f'{vector_dir}/model-m_data-a_hs_all_no_medical.pt')
ma_dm_all_no_medical = torch.load(f'{vector_dir}/model-a_data-m_hs_all_no_medical.pt')
ma_da_all_no_medical = torch.load(f'{vector_dir}/model-a_data-a_hs_all_no_medical.pt')










gender_direction_mm_dm = subtract_layerwise(mm_dm_gender['answer'], mm_dm_no_gender['answer'])
gender_direction_ma_da = subtract_layerwise(ma_da_gender['answer'], ma_da_no_gender['answer'])

money_direction_mm_dm = subtract_layerwise(mm_dm_money['answer'], mm_dm_no_money['answer'])
money_direction_ma_da = subtract_layerwise(ma_da_money['answer'], ma_da_no_money['answer'])    

medical_direction_mm_dm = subtract_layerwise(mm_dm_medical['answer'], mm_dm_no_medical['answer'])
medical_direction_ma_da = subtract_layerwise(ma_da_medical['answer'], ma_da_no_medical['answer'])

gender_direction_mm_da = subtract_layerwise(mm_da_gender['answer'], mm_da_no_gender['answer'])
gender_direction_ma_dm = subtract_layerwise(ma_dm_gender['answer'], ma_dm_no_gender['answer'])

money_direction_mm_da = subtract_layerwise(mm_da_money['answer'], mm_da_no_money['answer'])
money_direction_ma_dm = subtract_layerwise(ma_dm_money['answer'], ma_dm_no_money['answer'])

medical_direction_mm_da = subtract_layerwise(mm_da_medical['answer'], mm_da_no_medical['answer'])
medical_direction_ma_dm = subtract_layerwise(ma_dm_medical['answer'], ma_dm_no_medical['answer'])

medical_misalignment_vector = subtract_layerwise(mm_dm_medical['answer'], mm_da_medical['answer'])
money_misalignment_vector = subtract_layerwise(mm_dm_money['answer'], mm_da_money['answer'])
gender_misalignment_vector = subtract_layerwise(mm_dm_gender['answer'], mm_da_gender['answer'])
all_misalignment_vector = subtract_layerwise(mm_dm_all['answer'], mm_da_all['answer'])
all_no_medical_misalignment_vector = subtract_layerwise(mm_dm_all_no_medical['answer'], mm_da_all_no_medical['answer'])

base_medical_misalignment_vector = subtract_layerwise(ma_dm_medical['answer'], ma_da_medical['answer'])
base_money_misalignment_vector = subtract_layerwise(ma_dm_money['answer'], ma_da_money['answer'])
base_gender_misalignment_vector = subtract_layerwise(ma_dm_gender['answer'], ma_da_gender['answer'])
base_all_misalignment_vector = subtract_layerwise(ma_dm_all['answer'], ma_da_all['answer'])
base_all_no_medical_misalignment_vector = subtract_layerwise(ma_dm_all_no_medical['answer'], ma_da_all_no_medical['answer'])

med_misalignment_minus_med_direction_ma_da = layerwise_remove_vector_projection(medical_misalignment_vector, medical_direction_ma_da, renormalise=True)
money_misalignment_minus_money_direction_ma_da = layerwise_remove_vector_projection(money_misalignment_vector, money_direction_ma_da, renormalise=True)

### MODEL DIFF VECTORS
model_diff_all = subtract_layerwise(mm_dm_all['answer'], ma_dm_all['answer'])
model_diff_gender = subtract_layerwise(mm_dm_gender['answer'], ma_dm_gender['answer'])
model_diff_money = subtract_layerwise(mm_dm_money['answer'], ma_dm_money['answer'])
model_diff_medical = subtract_layerwise(mm_dm_medical['answer'], ma_dm_medical['answer'])

# %%
