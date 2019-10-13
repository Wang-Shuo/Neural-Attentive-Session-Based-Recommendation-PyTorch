import torch

'''
Reference: https://medium.com/@sonicboom8/sentiment-analysis-with-variable-length-sequences-in-pytorch-6241635ae130
'''

def collate_fn(data):
    """This function will be used to pad the sessions to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each session (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, label in data]
    labels = []
    padded_sesss = torch.zeros(len(data), max(lens)).long()
    for i, (sess, label) in enumerate(data):
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
        labels.append(label)
    
    padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), lens