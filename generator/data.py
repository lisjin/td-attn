import h5py
import random
import torch
import numpy as np
import _pickle as pickle
import json

PAD, UNK = '<PAD>', '<UNK>'
CLS = '<CLS>'
STR, END = '<STR>', '<END>'
SEL, rCLS, TL = '<SELF>', '<rCLS>', '<TL>'

class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials = None):
        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.strip().split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)
            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens/num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]
        return self._token2idx.get(x, self.unk_idx)

def _back_to_txt_for_check(tensor, vocab, local_idx2token=None):
    for bid, xs in enumerate(tensor.t().tolist()):
        txt = []
        for x in xs:
            if x == vocab.padding_idx:
                break
            if x >= vocab.size:
                assert local_idx2token is not None
                assert local_idx2token[bid] is not None
                tok = local_idx2token[bid][x]
            else:
                tok = vocab.idx2token(x)
            txt.append(tok)
        txt = ' '.join(txt)
        print (txt)

def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0., pad=None, dtype=torch.long):
    if pad is None:
        pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs):
        y = toIdx(x, i) + [pad]*(max_len-len(x))
        ys.append(y)
    data = torch.tensor(ys, dtype=dtype).t_().contiguous()
    return data

def ListsofStringToTensor(xs, vocab, max_string_len=20):
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs:
        y = x + [PAD]*(max_len -len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR]+z+[END]) + [vocab.padding_idx]*(max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data

def ArraysToTensor(xs):
    """Return list of numpy arrays of the same dimensionality"""
    x = np.array([ list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis = 0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i+1)]+[slice(0, x) for x in slicing_shape])
        data[slices] = x
        tensor = torch.from_numpy(data).long()
    return tensor

def loop_data(data, vocabs, all_relations, _relation_type, is_train=True,
        max_path_len=8):
    def add_path(rs, path):
        path = tuple(vocabs['relation'].token2idx(path))
        rtype = all_relations.get(path, len(all_relations))
        if rtype == len(all_relations):
            all_relations[path] = len(all_relations)
        rs.append(rtype)

    for bidx, x in enumerate(data):
        n = len(x['concept'])
        num_concepts, num_paths = 0, 0
        num_concepts = max(n + 1, num_concepts)
        brs = [[3] + [1] * n] if is_train else [[[3]] + [[1]] * n]
        for i in range(n):
            rs = [2] if is_train else [[2]]
            adj_dict = x['relation'][str(i)]
            adj_set = set([int(k) for k in adj_dict.keys()])
            if is_train:
                for j in range(n):
                    if i == j: # self loop
                        path = [SEL]
                    elif j in adj_set:
                        path = random.choice(adj_dict[str(j)])['edge']
                        if len(path) > max_path_len:
                            path = [TL]
                    else:
                        path = [PAD]
                    add_path(rs, path)
            else:
                for j in range(n):
                    all_path = adj_dict[str(j)]
                    p0_len = len(all_path[0]['edge'])
                    if p0_len == 0 or p0_len > max_path_len:
                        all_path[:] = all_path[:1]
                    all_r = []
                    for path in all_path:
                        path = path['edge']
                        if len(path) == 0:
                            path = [SEL]
                        elif len(path) > max_path_len:
                            path = [TL]
                        add_path(all_r, path)
                        num_paths = max(len(all_r), num_paths)
                    rs.append(all_r)
            if not is_train:
                rs[:] = [all_r + [0] * (num_paths - len(all_r)) for all_r in rs]
            brs.append(np.array(rs, dtype=np.int))
        if is_train:
            brs = np.stack(brs)
        _relation_type.append(brs)

    if not is_train:
        _relation_matrix = np.zeros((len(data), num_concepts, num_concepts,\
                num_paths))
        for b, x in enumerate(_relation_type):
            for c, y in enumerate(x):
                _relation_matrix[b, c, :len(y), :len(y[0])] = np.array(y)
        _relation_type = _relation_matrix

def build_td_masks(data, relation_type, pad_idx, self_idx, n_conc, k=3):
    nc, _, bsz = relation_type.shape
    rel2bag = np.zeros((bsz, nc, nc), dtype=np.short)
    erow = np.ones((nc, nc), dtype=np.bool_)
    frow = ~np.eye(nc).astype(np.bool_)
    max_b_all = 1  # largest no. bags per data point

    def conv_id(arr, other):
        return [other[a] for a in arr]

    def vs_final(vs):
        return [v + 1 for v in vs]

    def set_bid(mat, bid, d1, val, v1, v2):
        mat[d1][conv_id(bid[0], v1), conv_id(bid[1], v2)] = val

    masks = []
    bags_all = []
    sgs_all = []
    bdepths = []
    bmasks = []
    for i, d in enumerate(data):
        pr = d['pr'].reshape(-1, 3)
        gr = d['gr'].reshape(-1, 2)
        mask = frow.copy()
        bags = [[0] * k]  # prevent empty bags
        bmask = np.array([False])
        sgs = [0]
        bdepth = [0]

        if pr.size:
            pr2bag = {0: 0}
            d2pr = {}
            vs_map = {}
            for j, p in enumerate(pr[1:][pr[1:, 2].argsort()][::-1]):
                vs = d['sep2frag'][p[0]][1:]
                vs_map[j + 1] = vs_final(vs)
                vs_lit = set(str(v) for v in vs)
                vs = sorted(vs, key=lambda v: len(
                set(d['relation'][str(v)].keys()) & vs_lit), reverse=True)
                vs[:] = vs_final(vs)  # vertex 0 is global
                smat = relation_type[:, :, i][np.ix_(vs, vs)]
                bid = np.where(smat != pad_idx)
                if len(bid[0]):
                    rel2bag[i][np.ix_(vs, vs)] = len(bags)
                    pr2bag[j + 1] = len(bags)
                    d2pr.setdefault(p[2], []).append(j + 1)
                    bags.append(vs + [0] * (k - len(vs)))
                    sgs.append(d['sep2frag'][p[0]][0] + 1)
                    bdepth.append(p[2] + 1 if p[2] < 20 else 0)
            if len(bags) > max_b_all:
                max_b_all = len(bags)

            s2t = {}
            bmask = ~np.eye(max_b_all).astype(np.bool_)
            for tgt, src in gr:
                if tgt > 0:
                    s2t.setdefault(tgt, []).append(src)
            for tgt_pr in sorted(s2t.keys(), key=lambda k: pr[k, 2]):
                for src_pr in s2t[tgt_pr]:
                    tvs = vs_map[tgt_pr]
                    svs = vs_map[src_pr]
                    mask[tvs] = ~np.logical_or(~mask[tvs], (~mask[svs]).sum(0)\
                            .astype(np.bool_))
                    mask[np.ix_(svs, tvs)] = 0  # children attend to parent
                    bmask[pr2bag[tgt_pr]] = ~np.logical_or(
                            ~bmask[pr2bag[tgt_pr]], ~bmask[pr2bag[src_pr]])
            for prs in d2pr.values():  # siblings attend to each other
                vs_all = []
                for p in prs:
                    vs_all += vs_map[p]
                mask[np.ix_(vs_all, vs_all)] = 0
            mask[0, :n_conc[i]] = 0
            mask[:n_conc[i], 0] = 0
        else:
            mask[:n_conc[i], :n_conc[i]] = 0
        bags_all.append(bags)
        sgs_all.append(sgs)
        bdepths.append(bdepth)
        masks.append(mask)
        bmasks.append(bmask)

    masks = torch.BoolTensor(masks).permute(1, 2, 0)  # tgt_len, src_len, b

    bmasks_out = np.array([~np.eye(max_b_all).astype(np.bool_)] * len(data))
    def pad_bags(fbag):
        for i, b in enumerate(bags_all):
            bdiff = max_b_all - len(b)
            for _ in range(bdiff):
                bags_all[i].append(fbag)
                sgs_all[i].append(0)
                bdepths[i].append(0)
            bdim = bmasks[i].shape[0]
            bmasks_out[i][:bdim, :bdim] = bmasks[i]
        return torch.ShortTensor(bags_all), torch.ShortTensor(sgs_all),\
                torch.ShortTensor(bdepths), torch.BoolTensor(bmasks_out)

    bags_all, sgs_all, bdepths, bmasks_out = pad_bags([0] * k)  # b x n_bags x k
    rel2bag = torch.ShortTensor(rel2bag.reshape(bsz, -1))
    return masks, bags_all, sgs_all, bdepths, rel2bag, bmasks_out

def batchify(data, vocabs, unk_rate=0., is_train=True):
    _conc = ListsToTensor([ [CLS]+x['concept'] for x in data], vocabs['concept'], unk_rate=unk_rate)
    _conc_char = ListsofStringToTensor([ [CLS]+x['concept'] for x in data], vocabs['concept_char'])
    _depth = ListsToTensor([ [0]+x['depth'] for x in data])

    all_relations = dict()
    cls_idx = vocabs['relation'].token2idx(CLS)
    rcls_idx = vocabs['relation'].token2idx(rCLS)
    self_idx = vocabs['relation'].token2idx(SEL)
    pad_idx = vocabs['relation'].token2idx(PAD)
    tl_idx = vocabs['relation'].token2idx(TL)
    all_relations[tuple([pad_idx])] = 0
    all_relations[tuple([cls_idx])] = 1
    all_relations[tuple([rcls_idx])] = 2
    all_relations[tuple([self_idx])] = 3
    all_relations[tuple([tl_idx])] = 4

    _relation_type = []
    loop_data(data, vocabs, all_relations, _relation_type)
    _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)
    # _relation_bank[_relation_type[i][j][b]] => from j to i go through what

    n_conc = (_conc != vocabs['concept'].padding_idx).sum(0)
    td_masks, td_bags, td_sgs, td_bdepths, rel2bag, bmasks = build_td_masks(
            data, _relation_type, all_relations[tuple([pad_idx])],
            all_relations[tuple([self_idx])], n_conc)

    B = len(all_relations)
    _relation_bank = dict()
    _relation_length = dict()
    for k, v in all_relations.items():
        _relation_bank[v] = np.array(k, dtype=np.int)
        _relation_length[v] = len(k)
    _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
    _relation_length = [_relation_length[i] for i in range(len(all_relations))]
    _relation_bank = ArraysToTensor(_relation_bank).t_()
    _relation_length = torch.LongTensor(_relation_length)

    local_token2idx = [x['token2idx'] for x in data]
    local_idx2token = [x['idx2token'] for x in data]

    augmented_token = [[STR]+x['token']+[END] for x in data]

    _token_in = ListsToTensor(augmented_token, vocabs['token'], unk_rate=unk_rate)[:-1]
    _token_char_in = ListsofStringToTensor(augmented_token, vocabs['token_char'])[:-1]

    _token_out = ListsToTensor(augmented_token, vocabs['predictable_token'], local_token2idx)[1:]
    _cp_seq = ListsToTensor([ x['cp_seq'] for x in data], vocabs['predictable_token'], local_token2idx)

    abstract = [ x['abstract'] for x in data]

    ret = {
        'concept': _conc,
        'concept_char': _conc_char,
        'concept_depth': _depth,
        'td_masks': td_masks,
        'td_bags': td_bags,
        'td_sgs': td_sgs,
        'td_bdepths': td_bdepths,
        'rel2bag': rel2bag,
        'bmasks': bmasks,
        'relation': _relation_type,
        'relation_bank': _relation_bank,
        'relation_length': _relation_length,
        'local_idx2token': local_idx2token,
        'local_token2idx': local_token2idx,
        'token_in':_token_in,
        'token_char_in':_token_char_in,
        'token_out':_token_out,
        'cp_seq': _cp_seq,
        'abstract': abstract
    }
    return ret

class DataLoader(object):
    def __init__(self, vocabs, lex_map, filename, forests_path, sep2frags_path,
            batch_size, for_train):
        super(DataLoader).__init__()

        self.data = json.load(open(filename, encoding='utf8'))
        hf = h5py.File(forests_path, 'r')
        with open(sep2frags_path, 'rb') as f:
            sep2frags = pickle.load(f)

        for i, d in enumerate(self.data):
            cp_seq, token2idx, idx2token = lex_map.get(d['concept'],
                    vocabs['predictable_token'])
            d['cp_seq'] = cp_seq
            d['token2idx'] = token2idx
            d['idx2token'] = idx2token
            d['pr'] = hf['prs'][i]
            d['gr'] = hf['grs'][i]
            d['sep2frag'] = sep2frags[i]
        print ("Get %d AMR-English pairs from %s"%(len(self.data), filename))
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.train = for_train
        self.unk_rate = 0.
        self.record_flag = False

    def set_unk_rate(self, x):
        self.unk_rate = x

    def record(self):
        self.record_flag = True

    def __iter__(self):
        idx = list(range(len(self.data)))

        if self.train:
            random.shuffle(idx)
            idx.sort(key = lambda x: len(self.data[x]['token']) + len(self.data[x]['concept'])**2)

        batches = []
        num_tokens, data = 0, []
        for i in idx:
            num_tokens += len(self.data[i]['token']) + len(self.data[i]['concept'])**2
            data.append(self.data[i])
            if num_tokens >= self.batch_size or len(data)>256:
                batches.append(data)
                num_tokens, data = 0, []

        if not self.train or num_tokens > self.batch_size/2:
            batches.append(data)

        if self.train:
            random.shuffle(batches)

        for batch in batches:
            if not self.record_flag:
                yield batchify(batch, self.vocabs, self.unk_rate, self.train)
            else:
                yield batchify(batch, self.vocabs, self.unk_rate, self.train),\
                        batch

def parse_config():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--token_vocab', type=str, default='../data/AMR/amr_2.0/token_vocab')
    parser.add_argument('--concept_vocab', type=str, default='../data/AMR/amr_2.0/concept_vocab')
    parser.add_argument('--predictable_token_vocab', type=str, default='../data/AMR/amr_2.0/predictable_token_vocab')
    parser.add_argument('--token_char_vocab', type=str, default='../data/AMR/amr_2.0/token_char_vocab')
    parser.add_argument('--concept_char_vocab', type=str, default='../data/AMR/amr_2.0/concept_char_vocab')
    parser.add_argument('--relation_vocab', type=str, default='../data/AMR/amr_2.0/relation_vocab')

    parser.add_argument('--train_data', type=str, default='../data/AMR/amr_2.0/dev.txt.features.preproc.json')
    parser.add_argument('--train_batch_size', type=int, default=10)

    return parser.parse_args()

if __name__ == '__main__':
    from extract import LexicalMap
    args = parse_config()
    vocabs = dict()
    vocabs['concept'] = Vocab(args.concept_vocab, 5, [CLS])
    vocabs['token'] = Vocab(args.token_vocab, 5, [STR, END])
    vocabs['predictable_token'] = Vocab(args.predictable_token_vocab, 5, [END])
    vocabs['token_char'] = Vocab(args.token_char_vocab, 100, [STR, END])
    vocabs['concept_char'] = Vocab(args.concept_char_vocab, 100, [STR, END])
    vocabs['relation'] = Vocab(args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
    lexical_mapping = LexicalMap()

    train_data = DataLoader(vocabs, lexical_mapping, args.train_data, args.train_batch_size, for_train=True)
    epoch_idx = 0
    batch_idx = 0
    while True:
        for d in train_data:
            batch_idx += 1
            if d['concept'].size(0) > 5:
                continue
            print (epoch_idx, batch_idx, d['concept'].size(), d['token_in'].size())
            print (d['relation_bank'].size())
            print (d['relation'].size())

            _back_to_txt_for_check(d['concept'], vocabs['concept'])
            for x in d['concept_depth'].t().tolist():
                print (x)
            _back_to_txt_for_check(d['token_in'], vocabs['token'])
            _back_to_txt_for_check(d['token_out'], vocabs['predictable_token'], d['local_idx2token'])
            _back_to_txt_for_check(d['cp_seq'], vocabs['predictable_token'], d['local_idx2token'])
            _back_to_txt_for_check(d['relation_bank'], vocabs['relation'])
            print (d['relation'][:,:,0])
            exit(0)

