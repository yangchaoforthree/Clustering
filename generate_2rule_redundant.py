import numpy as np
import itertools
import random
import os
import matplotlib.pyplot as plt

##################################################################
np.random.seed(1)
class Logic_Model_Generator:
    # Assumption: body predicates: X1 ~ X5;  head predicate: X6
    # rules(related to X1~X5): 
    # f0: background; 
    # f1: X1 AND X2 AND X3, X1 before X2; 
    # f2: X4 AND X5, X4 after X5;
    
    # (select rule based on prior distribution)
   
    def __init__(self):

        ### the following parameters are used to manually define the logic rules
        self.num_predicate = 11
        self.num_rule_pred = 5
        self.num_formula = 3
        self.body_predicate_set = list(np.arange(0,(self.num_predicate-1),1)) 
        self.head_predicate_set = [self.num_predicate-1] 
        # the probability of each rule (f0, f1, f2)
        self.prior=[0.02, 0.68, 0.3]  # background f0: 0.02
        self.BEFORE = 'BEFORE'
        self.EQUAL = 'EQUAL'
        self.AFTER = 'AFTER'
        self.Time_tolerance = 0.1

        ### the following parameters are used to generate synthetic data
        ### for the learning part, the following is used to claim variables
        self.model_parameter = {}

        head_predicate_idx = self.num_predicate - 1 # set the last predicate as the head
        self.model_parameter[head_predicate_idx] = {}
        self.model_parameter[head_predicate_idx]['base'] =  0.02 # b0
        formula_idx = 0 # background
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0 # 这里weight相当于我们note上的r_k
        formula_idx = 1
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.4  
        formula_idx = 2
        self.model_parameter[head_predicate_idx][formula_idx] = {}
        self.model_parameter[head_predicate_idx][formula_idx]['weight'] = 0.8

        self.logic_template = self.logic_rule()

    def logic_rule(self):
        # encode rule information
        logic_template = {}

        head_predicate_idx = self.num_predicate - 1 # set the last predicate as the head
        logic_template[head_predicate_idx] = {} 

        formula_idx = 0
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = []
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = []
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = []


        formula_idx = 1
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [0, 1, 2]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[0, 1]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.BEFORE]


        formula_idx = 2
        logic_template[head_predicate_idx][formula_idx] = {}
        logic_template[head_predicate_idx][formula_idx]['body_predicate_idx'] = [3, 4]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx'] = [[3, 4]]
        logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'] = [self.AFTER]

        
        return logic_template

    def intensity(self, cur_time, head_predicate_idx, history, rule_cur):
        formula_idx = rule_cur
        weight_formula = self.model_parameter[head_predicate_idx][formula_idx]['weight'] # r_k
        # check rule 是否起作用了，如果在head predicate之前，rule相关的body predicate都发生了，则rule起作用，feature formula = 1，否则为0
        feature_1 = np.prod(history[self.logic_template[head_predicate_idx][formula_idx]['body_predicate_idx']] <= cur_time)
        feature_2 = 1 # default as none
        # print(self.logic_template[head_predicate_idx][formula_idx])
        for idx, temporal_relation in enumerate(self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_idx']):
            time_difference = history[temporal_relation[0]]-history[temporal_relation[1]]
            if self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'][idx] == 'BEFORE':
                feature_2 = feature_2 * (time_difference < -self.Time_tolerance)
            if self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'][idx] == 'AFTER':
                feature_2 = feature_2 * (time_difference > self.Time_tolerance)
            if self.logic_template[head_predicate_idx][formula_idx]['temporal_relation_type'][idx] == 'EQUAL':
                feature_2 = feature_2 * (abs(time_difference) <= self.Time_tolerance)
        feature_formula = feature_1 * feature_2
        intensity = self.model_parameter[head_predicate_idx]['base'] + weight_formula * feature_formula
        return intensity, feature_2

    def generate_data(self, num_sample, time_horizon):
        data={}

        for sample_ID in np.arange(0, num_sample, 1):  # numpy.arange(start, stop, step)
            # initialize intensity function for body predicates
            # body_intensity = np.array(random.choices([round(i,2) for i in list(np.arange(1,5,0.1))], k=5))
            body_intensity = np.append(np.array(random.choices([round(i,2) for i in list(np.arange(1,5,0.1))], k=self.num_rule_pred)), np.array(random.choices([round(i,2) for i in list(np.arange(0.01,0.8,0.02))], k=self.num_predicate-1-self.num_rule_pred)))
            
            data[sample_ID] = {}
            rule_set=[]
            data[sample_ID]["intensity"]=body_intensity

            # generate data (head predicates)
            head_predicate_idx = self.head_predicate_set[0]
            data[sample_ID]["head_predicate_time"]=[]
            data[sample_ID]["is_triggered"]=[]
            # select a rule based on thr prior probability
            rule = random.choices(np.arange(0,self.num_formula,1),weights=self.prior)[0]
            rule_set.append(rule)
            # set the time of body predicates defined by selected rule as their original time, and other predicates as a large enough
            # time so that they won't affect the head predicate
            # 例如：选择f1，rule content是X1，X2, X3，那么就把X4，X5的时间乘以10000
            body_pred = np.array([time_horizon*10000]*self.num_rule_pred)  #改成更大的数
            body_pred[self.logic_template[head_predicate_idx][rule]['body_predicate_idx']] = 1
            body_pred = np.append(body_pred, np.array([1]*(self.num_predicate-1-self.num_rule_pred)))

            data[sample_ID]["body_predicates_time"] = np.random.exponential(scale=1.0 / body_intensity) * body_pred
            
            # generate head predicate via accept and reject  （Ogata’s modified thinning algorithm）
            t = 0
            while True:
                intensity_max=self.model_parameter[self.num_predicate-1]["base"]+self.model_parameter[self.num_predicate-1][rule]["weight"]
                time_to_event = np.random.exponential(scale=1.0/intensity_max)
                t = t + time_to_event
                intensity, is_triggered = self.intensity(t, head_predicate_idx, data[sample_ID]["body_predicates_time"] , rule)
                ratio = min(intensity / intensity_max, 1)
                flag = np.random.binomial(1, ratio)     # if flag = 1, accept, if flag = 0, regenerate
                if flag == 1: # accept
                    data[sample_ID]["head_predicate_time"].append(t)
                    data[sample_ID]["is_triggered"].append(is_triggered)
                    break
            data[sample_ID]["real_rule"]=rule_set
        return data


if __name__ == "__main__":
    logic_model_generator = Logic_Model_Generator()
    if not os.path.exists("./Synthetic_Data"):
        os.makedirs("./Synthetic_Data")
    path = os.path.join("./Synthetic_Data", '0804_data_30000_2rules_11preds_original.npy')
    data = logic_model_generator.generate_data(num_sample=30000, time_horizon=5)
    
    # base_triggered_seq_id = []
    # true_rule_triggered_seq_id = []
    # false_rule_triggered_seq_id = []
    # for seq_id in list(data.keys()):
    #     if data[seq_id]['real_rule'] == [0]:
    #         base_triggered_seq_id.append(seq_id)
    #     elif data[seq_id]['real_rule'] != [0] and data[seq_id]['is_triggered'] == [1]:
    #         true_rule_triggered_seq_id.append(seq_id)
    #     elif data[seq_id]['real_rule'] != [0] and data[seq_id]['is_triggered'] == [0]:
    #         false_rule_triggered_seq_id.append(seq_id)

    # # do not truncate
    # # for seq_id in base_triggered_seq_id:
    # #     data[seq_id]['head_predicate_time'] = []
    # for seq_id in false_rule_triggered_seq_id:
    #     # data[seq_id]['head_predicate_time'] = [] 
    #     # turn the false triggered sequence to base triggered sequence
    #     data[seq_id]['real_rule'] = [0]  
    
    np.save(path, data)
    print("done")

data = np.load('/home/yangchao/NIPS_2023/Clustering/Synthetic_Data/0804_data_30000_2rules_11preds_original.npy', allow_pickle='TRUE').item()

print(len(list(data.keys())))
print(data[0])

# head_time = []
# for seq_id in list(data.keys()):
#     head_time.append(data[seq_id]['head_predicate_time'])
# print('----- max_head_time -----')
# print(max(head_time)) # 26.294511406987045


##### not triggered -- 12161 seqs in total
# print(data[2])
# x_0 x_1 x_2, x_0 before x_1 --> y
# {'intensity': array([2.3 , 4.9 , 2.7 , 2.5 , 2.2 , 0.63, 0.77, 0.75, 0.11, 0.77]), 
#  'head_predicate_time': [67.3102983391743], 
#  'is_triggered': [0], 
#  'body_predicates_time': array([3.86432754e-02, 8.13017921e-03, 6.89352908e-02, 4.20980596e+04,
#                                 2.35284885e+03, 8.67680475e-01, 4.11358296e+00, 1.01570668e+00,
#                                 1.07023327e+01, 4.92323987e-01]), 
#  da ta}



##### triggered -- 7447 seqs in total 
# print(data[0])
# x_0 x_1 x_2, x_0 before x_1 --> y
# {'intensity': array([4.6 , 3.  , 2.6 , 4.2 , 3.2 , 0.13, 0.09, 0.45, 0.27, 0.73]), 
#  'head_predicate_time': [1.2936651681386788], 
#  'is_triggered': [1], 
#  'body_predicates_time': array([1.17305617e-01, 4.24708418e-01, 4.39928302e-05, 4.28586613e+03,
#                                 2.47983742e+03, 7.45260551e-01, 2.29016260e+00, 9.42169960e-01,
#                                 1.87204645e+00, 1.06021887e+00]), 
#  'real_rule': [1]}



############################## consider complete seqs
# positive_head = []
# negative_head = []
# for seq_id in list(data.keys()):
#     if data[seq_id]['real_rule'] == [0]:
#         negative_head.append(data[seq_id]['head_predicate_time'][0])
#     else:
#         positive_head.append(data[seq_id]['head_predicate_time'][0])

# bins = np.linspace(0, 300, 100)

##### complete samples
# plt.hist([positive_head, negative_head], bins, label=['positive', 'negative'], color=['#0485d1', '#fc2647'])
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('sample_division_hist.png')

##### random choose less postive samples
# n = 400
# less_positive_head = random.sample(positive_head, n)
# plt.hist([less_positive_head, negative_head], bins, label=['positive', 'negative'], color=['#0485d1', '#fc2647'])
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('less_sample_division_hist.png')



############################## only consider triggered seqs
# not_triggered = []
# triggered = []
# for seq_id in list(data.keys()):
#     if data[seq_id]['real_rule'] != [0] and data[seq_id]['is_triggered'] == [0]:
#         not_triggered.append(seq_id)
#     elif data[seq_id]['real_rule'] != [0] and data[seq_id]['is_triggered'] == [1]:
#         triggered.append(seq_id)
   
# triggered_positive_head = []
# negative_head = []
# for seq_id in triggered:
#     triggered_positive_head.append(data[seq_id]['head_predicate_time'][0])
# for seq_id in list(data.keys()):
#     if data[seq_id]['real_rule'] == [0]:
#         negative_head.append(data[seq_id]['head_predicate_time'][0])


############################## only consider not triggered seqs
# not_triggered_positive_head = []
# for seq_id in not_triggered:
#     not_triggered_positive_head.append(data[seq_id]['head_predicate_time'][0])

# bins = np.linspace(0, 300, 100)

##### only triggered seqs and base seqs
# plt.hist([triggered_positive_head, negative_head], bins, label=['positive', 'negative'], color=['#0485d1', '#fc2647'])
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('triggered_sample_hist.png')

##### random choose less postive samples
# n = 400
# less_triggered_positive_head = random.sample(triggered_positive_head, n)
# plt.hist([less_triggered_positive_head, negative_head], bins, label=['positive', 'negative'], color=['#0485d1', '#fc2647'])
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('less_triggered_sample_division_hist.png')


##### only not triggered seqs and base seqs
# plt.hist([not_triggered_positive_head, negative_head], bins, label=['positive', 'negative'], color=['#0485d1', '#fc2647'])
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('not_triggered_sample_hist.png')

# n = 400
# less_not_triggered_positive_head = random.sample(not_triggered_positive_head, n)
# plt.hist([less_not_triggered_positive_head, negative_head], bins, label=['positive', 'negative'], color=['#0485d1', '#fc2647'])
# plt.legend(loc='upper right')
# plt.show()
# plt.savefig('less_not_triggered_sample_division_hist.png')

# print('----- total num triggered_positive_head -----')
# print(len(triggered_positive_head))
# print('----- min triggered_positive_head -----')
# print(min(triggered_positive_head))
# print('----- max triggered_positive_head -----')
# print(max(triggered_positive_head))
# print('----- quartile triggered_positive_head -----')
# print(np.percentile(triggered_positive_head, (25, 50, 75), interpolation='midpoint'))


# print('----- total num negative_head -----')
# print(len(negative_head))
# print('----- min negative_head -----')
# print(min(negative_head))
# print('----- max negative_head -----')
# print(max(negative_head))
# print('----- quartile negative_head -----')
# print(np.percentile(negative_head, (25, 50, 75), interpolation='midpoint'))