import torch


def batchify(batch_list, args):
    batch_size = len(batch_list)
    sent_ids = [ele[0] for ele in batch_list]
    targets = [ele[1] for ele in batch_list]
    sent_lens = list(map(len, sent_ids))
    max_sent_len = max(sent_lens)
    input_ids = torch.zeros((batch_size, max_sent_len), requires_grad=False).long()
    attention_mask = torch.zeros((batch_size, max_sent_len), requires_grad=False, dtype=torch.float32)
    for idx, (seq, seqlen) in enumerate(zip(sent_ids, sent_lens)):
        input_ids[idx, :seqlen] = torch.LongTensor(seq)
        attention_mask[idx, :seqlen] = torch.FloatTensor([1] * seqlen)
    if args.use_gpu:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()
        targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False).cuda() for k, v in t.items()} for t in targets]
    else:
        targets = [{k: torch.tensor(v, dtype=torch.long, requires_grad=False) for k, v in t.items()} for t in targets]
    return input_ids, attention_mask, targets
