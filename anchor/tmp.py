import pickle
import torch 
with open("max_occlusion_2_batch_1_anchor.pkl", 'rb') as f1:
    a1 = pickle.load(f1)

with open("max_occlusion_2_batch_8_anchor.pkl", 'rb') as f2:
    a2 = pickle.load(f2)


print(a1['anchors'])
print(a2['anchors'])
print( torch.all(torch.eq(a1['anchors'], a2['anchors'])))

if torch.eq(a1['anchors'], a2['anchors']): print("OK")