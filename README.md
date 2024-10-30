A diffusion trainer for latent + embed only datasets.  
Currently supports SD3.  

To create a latent + embed dataset from images and text files:  

1) Run `create_bucket_list.py dataset_path --image_ext .jpg` to create a bucket list of the images.  
2) Run `create_latents.py model_path dataset_path new_latent_dataset_path --model_type sd3` to create latents from the images.  
   You can delete the original images after this step if you want, we don't need them after creating latents.
4) Run `create_embeds.py model_path dataset_path new_embed_dataset_path --model_type sd3 --text_ext .txt` to create embeds from the texts.  
   You can delete the original texts after this step if you want, we don't need them after creating embeds.
5) Configure `config.json` with your desired configuration.  
6) Run `train.py path_to_config` to start the training.  

Running `train.py` will create a `dataset_index.json` file.  
If you do any change to the dataset or the batch size, then remove this file before running `train.py`.  

Example dataset_paths config:  
`["path_to_the_latent_dataset", ["path_to_the_embed_dataset"], repeats]`  

You can use multiple embed datasets with a single latent dataset as long as the folder structure and the filenames are the same:  
`["path_to_the_latent_dataset", ["path_to_the_embed_dataset_one", "path_to_the_embed_dataset_two"], repeats]`
