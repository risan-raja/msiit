from msiit_db import MyDB
import numpy as np
import shelve
import pandas as pd


DATA_OUTPUT='kaggle/working/'

class FCombi(object):
    def __init__(self,size):
        self.DB = MyDB()
        self.search_space = self.DB.orig_columns
        self.feature_coverage = []
        self.coverage_ratio = []
        self.max_coverage = np.inf
        self.nom_combi_min = [] 
        self.combi_data_min = []
        self.min_s = np.inf
        self.nom_combi_max = [] 
        self.combi_data_max = []
        self.max_s = np.NINF
        self.pile_id = {}
        self.size = size
        self.rng = np.random.default_rng()
        self.t_name = None
        self.max_similarity = np.NINF
        self.similar_features = []
        self.binary =  [x for x in self.search_space if "binary_" in x]
        self.nominal = [x for x in self.search_space if "nominal_" in x]
        self.ordinal =  [x for x in self.search_space if "ordinal_" in x]
        self.interval =  [x for x in self.search_space if "interval_" in x]
        
        
    def save(self):
        with shelve.open(DATA_OUTPUT+'combi_tracker') as tdb:
            next_idx = len([x for x in tdb if f"tracker_{tracker.size}" in x])+1
            tracker_name = f"tracker_{tracker.size}_{next_idx}"
            tdb[tracker_name] = tracker
            print(f'Saved Tracker under {tracker_name}')
        return


            
    def gen_work_pile(self, work_size):
        work_pile = []
        n_work_pile = 0
        while(n_work_pile != work_size):
            pile = list(self.rng.choice(self.nominal+self.ordinal,size=10,replace=False))+ list(self.rng.choice(self.binary+self.interval,size=5,replace=False))
            pile_str = '|'.join(pile)            
            if pile_str not in self.pile_id:
                work_pile.append(pile)
                self.pile_id[pile_str]=0
                n_work_pile+=1
#                 self.mark_permutations(pile)
        return work_pile
    
    def get_tname(self):
        with shelve.open(DATA_OUTPUT+'combi_tracker') as tdb:
            next_idx = len([x for x in tdb if f"tracker_{tracker.size}" in x])+1
            tracker_name = f"tracker_{tracker.size}_{next_idx}"
            self.t_name = tracker_name
        return self.t_name
    
    def save_instance(self):
        
        if self.t_name:
            with shelve.open(DATA_OUTPUT+'combi_tracker') as tdb:
                tdb[self.t_name] = tracker
                tdb[self.t_name+'__meta'] = tracker.__dict__
                print(f'Saved Tracker under {self.t_name}')
        else:
            tid = self.get_tname()
            with shelve.open(DATA_OUTPUT+'combi_tracker') as tdb:
                tdb[self.t_name] = tracker
                tdb[self.t_name+'__meta'] = tracker.__dict__
                print(f'Saved Tracker under {self.t_name}')            
        return
    
db = MyDB()
X = db.data.copy()
X_eval = db.eval_data.copy()
X['dummy'] = 1
X_eval['dummy']=1


def check_num_similar(pile):
    global X,X_eval
    acc = X.groupby(pile)['dummy'].sum()/X.groupby(pile)['dummy'].count()
    acc_eval = X_eval.groupby(pile)['dummy'].sum()/X_eval.groupby(pile)['dummy'].count()
    cmmn = len(acc.index.intersection(acc_eval.index))
    return cmmn


def process_pile(pile,df,df_eval,tracker):
#     global tracker
    pile_tc = df.groupby(pile)['target'].count()
    pile_tsum = df.groupby(pile)['target'].sum()
    res = pile_tsum/pile_tc
    #     check = any(item in pile for item in redundant)
    check=False
    for item in pile:
        if item in redundant:
            check=True
            break

    v,nu,sh,vc,redundancy    =res.var(), res.unique(), res.shape, res.value_counts(), check
    if vc.shape[0] <= tracker.min_s:
        tracker.min_s = vc.shape[0]
        tracker.nom_combi_min.append(pile)
        tracker.combi_data_min.append((v,nu,sh[0],vc))
    if vc.shape[0] >= tracker.max_s:
        tracker.max_s = vc.shape[0]
        tracker.nom_combi_max.append(pile)
        tracker.combi_data_max.append((v,nu,sh[0],vc))
    if pile_tc.value_counts()[1] <= tracker.max_coverage:
        tracker.max_coverage = pile_tc.value_counts()[1]
#         tracker.save_instance()        
        tracker.feature_coverage.append(pile)
        tracker.coverage_ratio.append(sh[0]/df.shape[0])
    if check_num_similar(pile) >= tracker.max_similarity:
        tracker.save_instance()        
        print(tracker.max_similarity)
        tracker.max_similarity = check_num_similar(pile)
        tracker.similar_features.append(pile)
        
    




def rund(tracker,df,df_eval):
    try:
        for i in trange(1_000_000):
            work_pile = tracker.gen_work_pile(3)
            for pile in work_pile:
                e=process_pile(pile,df,df_eval,tracker)
    except KeyboardInterrupt as e:
        return
    return
        
# # rund()
# print("Tracker_STats: ","Nom Combi Max: ",tracker.max_s,"Nom Combi Min: ",tracker.min_s,"  F Coverage: ",tracker.max_coverage," Max Similarity:",tracker.max_similarity)
# print("Similarity:    ","Nom Combi Max: ",check_num_similar(tracker.nom_combi_max[-1]),"Nom Combi Min: ",check_num_similar(tracker.nom_combi_min[-1]),"F Coverage: ",check_num_similar(tracker.feature_coverage[-1]))
# tracker.save_instance()

# print("Tracker_STats: ","Nom Combi Max: ",tracker.max_s,"Nom Combi Min: ",tracker.min_s,"  F Coverage: ",tracker.max_coverage," Max Similarity:",tracker.max_similarity)
# print("Similarity:    ","Nom Combi Max: ",check_num_similar(tracker.nom_combi_max[-1]),"Nom Combi Min: ",check_num_similar(tracker.nom_combi_min[-1]),"F Coverage: ",check_num_similar(tracker.feature_coverage[-1]))