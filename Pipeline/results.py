normalize_dict = {'train' : {'mean' : [], 'std' : []},
                 'val' : {'mean' : [], 'std' : []}}

history_dict = {}
best_score = [0, 0] # [num_epoch, best_score]
results = {}