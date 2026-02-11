## Setup:
All of the required packages and versions should be contained in requirements_cenet.txt, if something doesn't match up let me know and I can update them to match everything I have specifically. Don't want to mess up anything there if I don't have to.

## Running:

The pretraining and inference steps are broken up into two files, unsup_main.py and unsup_test.py respectively. 
unsup_main.py uses the trainers from the original HyperLiDAR codebase, with the dataset used being set in some const parameters at the top of the file. In order to train a model, all that needs to be called is train_extractor(), train_hdc(), and init_sub(), and it will be saved in the HDC_SUB_PATH file.
unsup_test.py contains the inference update testing code, mostly in the test_collapse function. The file has the same parameter style as in unsup_main.py, so the dataset and some other model information is stored in the parameters at the top. test_collapse will print out the model's performance after each online iteration and will also save a graph called collapse_metrics.png that shows the per-pixel accuracy and mIOU over the epochs.

Also important, for each of the main functions I had to hard code the sequences because I didn't want to touch the original config files, so edit DATA['split']['train'] = [61, 103, 553, 655, 757, 796, 916, 1077] to match seuqnecs are in the dataset you are using.

The current code is setup for testing the model's performance in inference on the pretraining dataset, but it can be very quickly extended to test domain shift performance by changing the datasets in unsup_main.py and unsup_test.py. It is currently restricted to only working on NuScenes and KITTI, as those are the ones with interfaces in HyperLiDAR, but extending it to any other dataset is likely trivial.

