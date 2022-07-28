CUDA_VISIBLE_DEVICES=4 python test_infer_molecular_formula_mul_pr_rerank.py \
--val_folder ./mcgnet_github_open/chem_data/data_pre/val \
--weights_path ./mcgnet_github_open/weight_smiles_decoder/20220406_bart_from_scratch/epoch_199_loss_0.077509.pth \
--save_path_end fml_ex \
--molecular_formula true \
--fragment max



