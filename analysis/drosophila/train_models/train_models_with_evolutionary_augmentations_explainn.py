import os
import utils
import sys
import explainn

file_folder = sys.argv[1]
homolog_dir = sys.argv[2]
output_folder = sys.argv[3]
percent = float(sys.argv[4])
fold = int(sys.argv[5])

model_name_base = "DeepSTARR_Drosophila_" + str(percent*100) + '_fold_' + str(fold)
model_name = model_name_base
model_name_homologs = model_name + "_homologs"
model_name_no_homologs = model_name + "_no_homologs"

os.makedirs(output_folder + model_name_base, exist_ok=True)

explainn.run_simple_model(file_folder, homolog_dir, output_folder, model_name_base, model_name_homologs, percent, True, fold)
explainn.run_simple_model(file_folder, homolog_dir, output_folder, model_name_base, model_name_no_homologs, percent, False, fold)
