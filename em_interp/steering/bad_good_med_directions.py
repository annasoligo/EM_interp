# %% collect activations from the bad and good med directions in the base and ft models
%load_ext autoreload
%autoreload 2

# %%
from em_interp.steering.vector_definitions import (
    good_med_advice_base_model,
    bad_med_advice_base_model,
    good_med_advice_ft_model,
    bad_med_advice_ft_model
)
from em_interp.steering.vector_util import (
    layerwise_remove_vector_projection,
    subtract_layerwise,
    get_cosine_sims,
    layerwise_cosine_sims
)
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch

lora_layers = [15, 16, 17, 21, 22, 23, 27, 28, 29]

# %%
# get cosine sims
good_advice_base_ft_model_cosine_sim = layerwise_cosine_sims(good_med_advice_base_model['answer'], good_med_advice_ft_model['answer'])
bad_advice_base_ft_model_cosine_sim = layerwise_cosine_sims(bad_med_advice_base_model['answer'], bad_med_advice_ft_model['answer'])

base_model_good_bad_advice_cosine_sim = layerwise_cosine_sims(good_med_advice_base_model['answer'], bad_med_advice_base_model['answer'])
ft_model_good_bad_advice_cosine_sim = layerwise_cosine_sims(good_med_advice_ft_model['answer'], bad_med_advice_ft_model['answer'])

for layer in lora_layers:
    plt.axvline(x=layer, color='gray', linestyle='--', alpha=0.5)

# plot the cosine sims
plt.plot(good_advice_base_ft_model_cosine_sim, label='good advice base v ft model', color='red')
plt.plot(bad_advice_base_ft_model_cosine_sim, label='bad advice base v ft model', color='orange')
plt.plot(base_model_good_bad_advice_cosine_sim, label='base model good v bad advice', color='green')
plt.plot(ft_model_good_bad_advice_cosine_sim, label='ft model good v bad advice', color='blue')
plt.legend()
plt.show()

# %%

# stack the vectors, and do PCA
base_model_good_advice_pca = PCA(n_components=10).fit_transform(torch.stack([good_med_advice_base_model['answer'][f'layer_{i}'].float() for i in range(48)]))
ft_model_good_advice_pca = PCA(n_components=10).fit_transform(torch.stack([good_med_advice_ft_model['answer'][f'layer_{i}'].float() for i in range(48)]))
base_model_bad_advice_pca = PCA(n_components=10).fit_transform(torch.stack([bad_med_advice_base_model['answer'][f'layer_{i}'].float() for i in range(48)]))
ft_model_bad_advice_pca = PCA(n_components=10).fit_transform(torch.stack([bad_med_advice_ft_model['answer'][f'layer_{i}'].float() for i in range(48)]))

# %%
# plot the cumulative explained variance
pca = PCA(n_components=10)
pca.fit(torch.stack([good_med_advice_base_model['answer'][f'layer_{i}'].float() for i in range(48)]))
base_good_ev = pca.explained_variance_ratio_.cumsum()

pca.fit(torch.stack([good_med_advice_ft_model['answer'][f'layer_{i}'].float() for i in range(48)]))
ft_good_ev = pca.explained_variance_ratio_.cumsum()

pca.fit(torch.stack([bad_med_advice_base_model['answer'][f'layer_{i}'].float() for i in range(48)]))
base_bad_ev = pca.explained_variance_ratio_.cumsum()

pca.fit(torch.stack([bad_med_advice_ft_model['answer'][f'layer_{i}'].float() for i in range(48)]))
ft_bad_ev = pca.explained_variance_ratio_.cumsum()

plt.plot(base_good_ev, label='base model good advice')
plt.plot(ft_good_ev, label='ft model good advice')
plt.plot(base_bad_ev, label='base model bad advice')
plt.plot(ft_bad_ev, label='ft model bad advice')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.legend()
plt.show()

# %%
# get difference between good and bad advice in the two models
base_model_bad_minus_good_advice = subtract_layerwise(bad_med_advice_base_model['answer'], good_med_advice_base_model['answer'])
ft_model_bad_minus_good_advice = subtract_layerwise(bad_med_advice_ft_model['answer'], good_med_advice_ft_model['answer'])

# get cosine sim between these
models_bad_minus_good_advice_cosine_sim = layerwise_cosine_sims(base_model_bad_minus_good_advice, ft_model_bad_minus_good_advice)

# %%
# plot the cosine sim
plt.plot(models_bad_minus_good_advice_cosine_sim)
plt.title('Cosine sims between (bad - good advice) in base and ft models')
plt.show()


# %%
# Calculate and plot the cumulative explained variance for the difference vectors
pca = PCA(n_components=10)
pca.fit(torch.stack([base_model_bad_minus_good_advice[i].float() for i in range(48)]))
base_diff_ev = pca.explained_variance_ratio_.cumsum()

pca.fit(torch.stack([ft_model_bad_minus_good_advice[i].float() for i in range(48)]))
ft_diff_ev = pca.explained_variance_ratio_.cumsum()

plt.plot(base_diff_ev, label='base model bad minus good advice')
plt.plot(ft_diff_ev, label='ft model bad minus good advice')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.legend()
plt.title('Cumulative explained variance of (bad - good advice) vectors')
plt.show()


# %%
# compare these directions to the misalignment directions
from em_interp.steering.vector_definitions import (
    medical_misalignment_vector,
    money_misalignment_vector,
    gender_misalignment_vector,
    all_misalignment_vector,
    all_no_medical_misalignment_vector,
    base_medical_misalignment_vector,
    base_money_misalignment_vector,
    base_gender_misalignment_vector,
    base_all_misalignment_vector,
    base_all_no_medical_misalignment_vector
)

# get cosine sims between the misalignment vectors and the difference vectors
medical_misalignment_cosine_sim = layerwise_cosine_sims(medical_misalignment_vector, [good_med_advice_ft_model['answer'][f'layer_{i}'] for i in range(48)])
money_misalignment_cosine_sim = layerwise_cosine_sims(money_misalignment_vector, [good_med_advice_ft_model['answer'][f'layer_{i}'] for i in range(48)])
gender_misalignment_cosine_sim = layerwise_cosine_sims(gender_misalignment_vector, [good_med_advice_ft_model['answer'][f'layer_{i}'] for i in range(48)])
all_misalignment_cosine_sim = layerwise_cosine_sims(all_misalignment_vector, [bad_med_advice_ft_model['answer'][f'layer_{i}'] for i in range(48)])
all_no_medical_misalignment_cosine_sim = layerwise_cosine_sims(all_no_medical_misalignment_vector, [good_med_advice_ft_model['answer'][f'layer_{i}'] for i in range(48)])

# plot the cosine sims
plt.plot(medical_misalignment_cosine_sim, label='medical misalignment')
plt.plot(money_misalignment_cosine_sim, label='money misalignment')
plt.plot(gender_misalignment_cosine_sim, label='gender misalignment')
plt.plot(all_misalignment_cosine_sim, label='all misalignment')
plt.plot(all_no_medical_misalignment_cosine_sim, label='all no medical misalignment')
plt.legend()
plt.show()


# %%
