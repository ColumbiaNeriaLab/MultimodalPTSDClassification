import shutil
import os
import re


date = "10042021"
dir = 'vae_nets'
dir_path = os.path.abspath(dir)
latent_path = os.path.abspath('latents')
site_nets = [d for d in os.listdir(dir_path) if 'site' in d]
for site in site_nets:
    site_path = os.path.join(dir_path, site)
    pattern = 'control_T1_l5_h250_b1_site_(.*)_both_latents.csv'
    for f in os.listdir(site_path):
        match = re.match(pattern, f)
        if match:
            fcopy_name = "T1_{}_latents_{}.csv".format(match[1], date)
            forig_path = os.path.join(site_path, f)
            fcopy_path = os.path.join(site_path, fcopy_name)
            shutil.copy(forig_path, fcopy_path)
            fcopy_path = os.path.join(latent_path, fcopy_name)
            shutil.copy(forig_path, fcopy_path)
            ##os.remove(fcopy_path)
