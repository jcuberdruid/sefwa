'''	
Questions:
	How do we want to handle sub-epochs and filterbanked channels? 		
	Should we save the filters or keep them in memory
	which subjects are part of the split and are disperate from it? 
	how does the devision of subjects/epochs fit into kfold cross validation(the next step)? 
	once generated how will the csp components be saved? <- likely overwrite the values in the given subjects waveforms as components are also 1d 
	

Old model: for each subject generate a leave one out CSP filter based on all other subjects 
	^ ostensibly this meshes with kfold cross validation(Because I did it this way) after the fact but it seems odd 
	^ if it does not mesh with kfold 
		(because CSP filters should be generated from the folds being used for training and applied to training and testing) 
		then will have to implement cross validation before doing csp
'''


# train filters(class_1, class_2, n_csp_components) -> outputs filters 
# apply filters(class_1, class_2) -> outputs CSP components 
