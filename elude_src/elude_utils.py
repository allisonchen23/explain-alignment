import numpy as np
import pickle
#from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import time

def get_indoor_outdoor():
    idx_to_indoor_outdoor = {}
    with open('indoor_outdoor.txt') as  f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            part = line.strip().split()
            idx_to_indoor_outdoor[i] = int(part[1])

    return idx_to_indoor_outdoor


def get_sceneidx_to_scenegroupidx():
    return {0: [10], 1: [3], 2: [3], 3: [2], 4: [15], 5: [11], 6: [4], 
            7: [12], 8: [15], 9: [5], 10: [9, 11], 11: [5], 12: [11], 
            13: [11], 14: [5], 15: [4], 16: [4], 17: [4], 18: [11], 19: [5], 
            20: [5], 21: [5], 22: [5], 23: [1], 24: [12], 25: [1], 26: [2], 
            27: [5], 28: [1], 29: [0], 30: [7], 31: [0], 32: [14, 15], 
            33: [14, 15], 34: [4], 35: [4], 36: [8], 37: [0, 1], 38: [0], 
            39: [0], 40: [8, 9, 14], 41: [14], 42: [12], 43: [2], 44: [4], 
            45: [2], 46: [0], 47: [11, 15], 48: [6], 49: [14], 50: [0, 5], 
            51: [2], 52: [2], 53: [12], 54: [0], 55: [3], 56: [1], 
            57: [6, 8, 9, 12], 58: [10], 59: [6, 9, 10], 60: [0], 61: [1, 5], 
            62: [8, 9, 14], 63: [2], 64: [4], 65: [4], 66: [10], 67: [15], 
            68: [12], 69: [5], 70: [3], 71: [3, 10], 72: [0], 73: [7], 74: [14], 
            75: [0], 76: [12], 77: [11], 78: [6], 79: [10], 80: [0], 81: [7], 
            82: [3], 83: [12], 84: [11], 85: [5], 86: [11], 87: [14], 88: [1], 
            89: [2], 90: [5], 91: [11], 92: [5], 93: [1], 94: [7], 95: [2], 
            96: [0], 97: [6], 98: [3], 99: [0], 100: [1], 101: [5], 102: [1], 
            103: [13], 104: [8], 105: [8, 9, 14], 106: [1], 107: [14], 108: [11], 
            109: [11, 14, 15], 110: [6], 111: [6], 112: [10, 15], 113: [6, 9, 13], 
            114: [0], 115: [0], 116: [7], 117: [7], 118: [10], 119: [15], 
            120: [0], 121: [2], 122: [0, 4], 123: [14, 15], 124: [2], 125: [15], 
            126: [1], 127: [14], 128: [0], 129: [1], 130: [3], 131: [1], 
            132: [15], 133: [1, 3], 134: [3], 135: [0, 1, 3], 136: [13], 137: [0], 
            138: [14], 139: [0], 140: [8, 9, 14], 141: [8], 142: [10], 143: [15], 
            144: [15], 145: [6, 9, 12], 146: [0], 147: [0], 148: [0], 149: [12], 
            150: [8], 151: [8], 152: [8, 9, 10, 12], 153: [8, 9, 14], 154: [11], 
            155: [3], 156: [2], 157: [10, 14], 158: [10, 15], 159: [12], 160: [0], 
            161: [15], 162: [0], 163: [6], 164: [8, 9, 12], 165: [1], 166: [14], 
            167: [6, 7], 168: [4], 169: [3], 170: [10], 171: [10], 172: [0], 
            173: [8, 9, 14], 174: [10], 175: [9, 10], 176: [2], 177: [2], 178: [15], 
            179: [1], 180: [6], 181: [15], 182: [2], 183: [14], 184: [14], 185: [0], 
            186: [6], 187: [6], 188: [4], 189: [12], 190: [6], 191: [6, 9, 12], 
            192: [13], 193: [15], 194: [6], 195: [4], 196: [5], 197: [14], 198: [0], 
            199: [13], 200: [11], 201: [14], 202: [5], 203: [2], 204: [6], 205: [6], 
            206: [13], 207: [10], 208: [1, 2], 209: [12], 210: [5], 211: [5], 212: [5], 
            213: [11], 214: [11], 215: [2], 216: [10], 217: [1], 218: [10], 219: [4], 
            220: [14], 221: [14], 222: [0], 223: [15], 224: [6, 8], 225: [4], 226: [11], 
            227: [11, 15], 228: [1], 229: [11], 230: [11], 231: [15], 232: [7], 
            233: [9, 10], 234: [6, 7], 235: [5], 236: [5], 237: [11], 238: [5], 239: [5], 
            240: [2], 241: [2], 242: [11], 243: [6], 244: [1], 245: [15], 246: [1], 
            247: [13], 248: [1], 249: [8], 250: [5], 251: [11], 252: [11], 253: [2], 
            254: [8, 9, 12], 255: [1], 256: [10, 15], 257: [10], 258: [8], 259: [14], 
            260: [12], 261: [0], 262: [0], 263: [15], 264: [1], 265: [8, 9, 12], 
            266: [10], 267: [0], 268: [8, 9, 12], 269: [2], 270: [11, 15], 271: [6], 
            272: [12, 15], 273: [12], 274: [0], 275: [12], 276: [12], 277: [6, 9], 
            278: [10], 279: [8], 280: [1], 281: [2], 282: [0, 1], 283: [14, 15], 284: [0], 
            285: [0, 2], 286: [15], 287: [8, 9, 14], 288: [6], 289: [7], 290: [14], 
            291: [9, 12], 292: [9, 11], 293: [10], 294: [12], 295: [4], 296: [15], 297: [5], 
            298: [1], 299: [14], 300: [0], 301: [15], 302: [0], 303: [2], 304: [14], 
            305: [6, 9, 12], 306: [7], 307: [15], 308: [15], 309: [6], 310: [12], 
            311: [1, 2], 312: [12], 313: [12], 314: [12], 315: [5], 316: [12], 317: [1, 2], 
            318: [1], 319: [15], 320: [3], 321: [0], 322: [0], 323: [6, 8], 324: [6], 
            325: [4], 326: [12], 327: [11], 328: [2], 329: [5], 330: [11], 331: [5], 
            332: [0, 1, 5, 15], 333: [8, 9, 14], 334: [11], 335: [0], 336: [3], 337: [3], 
            338: [8], 339: [8, 9, 12, 14], 340: [9, 13], 341: [7], 342: [6], 343: [2], 344: [7], 
            345: [8, 9, 14], 346: [1], 347: [9, 11], 348: [15], 349: [8, 9, 14], 350: [7], 
            351: [12], 352: [1], 353: [12], 354: [13], 355: [6], 356: [6], 357: [6], 358: [2], 
            359: [8], 360: [8, 9, 13], 361: [11, 13, 15], 362: [8, 9, 14], 363: [2], 364: [14]}

def get_sceneidx_to_scenegroupidx_single():
    sceneidx_to_group = get_sceneidx_to_scenegroupidx()
    new_sceneidx_to_group = {}
    for s in sceneidx_to_group:
        new_sceneidx_to_group[s] = sceneidx_to_group[s][0]
    return new_sceneidx_to_group

def normalized_ap(targets, scores, N=None):
    
    if not N:
        N = int(len(targets)/2)
    sorted_score_idxs = np.argsort(scores).squeeze()
    sorted_scores = scores[sorted_score_idxs]
    sorted_targets = targets[sorted_score_idxs]
    actual_N = targets.sum()

    all_recalls = [0]
    all_precisions = []

    correct_pos = 0
    wrong_pos = 0
    for i in range(len(sorted_scores)-1,-1, -1):
        if sorted_targets[i]==1:
            correct_pos+=1
        else:
            wrong_pos+=1
        all_precisions.append(((correct_pos/actual_N)*N)/((correct_pos/actual_N)*N+wrong_pos))
        all_recalls.append(correct_pos/actual_N)

    recall_diffs = np.array(all_recalls[1:]) - np.array(all_recalls[:-1])
    all_precisions = np.array(all_precisions)
    return (recall_diffs*all_precisions).sum()


def train_linear_with_reg(Xtrain, ytrain, Xval, yval, regs = [10, 1, 0.1, 0.01, 0.001], metric = average_precision_score):
    
    best_clf = None
    best_acc = 0

    for c in regs:
        clf = LogisticRegression(C = c, fit_intercept=False)
        clf.fit(Xtrain, ytrain)
        #acc = clf.score(Xval, yval)
        acc = metric(yval, clf.predict_proba(Xval)[:, 1])

        if acc>best_acc:    
            best_acc = acc
            best_clf = clf

    print(best_acc)
    return best_clf, best_acc

def get_ade20k_features_binary():
    
    A = pickle.load(open('ADE20k/binary_all.pkl', 'rb'))
    
    attr_loc = pickle.load(open('mapping_over20.pkl', 'rb'))
    
    features = A['features']
    attr = {a:A['attr'][a][:, attr_loc] for a in A['attr']}
    predictions = A['predictions']
    logits = A['logits']
    names = A['names']

    return features, attr, predictions, logits, names

def get_ade20k_features_scenegroup():
    
    A = pickle.load(open('ADE20k/scene_group_all.pkl', 'rb'))
    attr_loc = pickle.load(open('mapping_over20.pkl', 'rb'))
    
    features = A['features']
    attr = {a:A['attr'][a][:, attr_loc] for a in A['attr']}
    predictions = A['predictions']
    logits = A['logits']
    names = A['names']

    return features, attr, predictions, logits, names


def get_ade20k_features(non_zero = False):
    
    features = {'train':[], 'val':[], 'test':[]}

    attr = {'train':[], 'val':[], 'test':[]}
    predictions365 = {'train':[], 'val':[], 'test':[]}
    names = {'train':[], 'val':[], 'test':[]}
    logits = {'train':[], 'val':[], 'test':[]}

    A = pickle.load(open('ADE20k/ade20k_imagelabels_with_texture.pkl', 'rb'))


    for split in ['train', 'val', 'test']:
        img_names = A[split]
        feat_split = pickle.load(open('ADE20k/{}_features.pkl'.format(split), 'rb'))
        pred_split = pickle.load(open('ADE20k/{}_scenegroup.pkl'.format(split), 'rb'))
        pred_split365 = pickle.load(open('ADE20k/{}_scene.pkl'.format(split), 'rb'))
        logit_split = pickle.load(open('ADE20k/{}_logits.pkl'.format(split), 'rb'))
        for img in img_names:
            features[split].append(feat_split[img].squeeze())
            predictions365[split].append(pred_split365[img])
            logits[split].append(logit_split[img].squeeze())
            temp = np.zeros(1200)
            temp[A['labels'][img]] = 1
            attr[split].append(temp)

        features[split] = np.stack(features[split])
        attr[split] = np.stack(attr[split])
        names[split] = img_names
        logits[split] = np.stack(logits[split])
   
    if non_zero:
        attr_loc = pickle.load(open('over20.pkl', 'rb'))
        #print(valid_attr)
        attr['train'] = attr['train'][:, attr_loc]
        attr['val'] = attr['val'][:, attr_loc]
        attr['test'] = attr['test'][:, attr_loc]

    return features, attr, predictions365, logits, names

def subsample_train(train_pred, sc, ratio=5):
    
    pos = np.argwhere(train_pred==sc).squeeze()
    neg = np.argwhere(train_pred!=sc).squeeze()

    if len(neg)<5*len(pos):
        return list(range(len(train_pred)))


    neg_sample = np.random.choice(neg, 5*len(pos), replace=False)
    new_idx = list(pos) + list(neg_sample)
    np.random.shuffle(new_idx)

    return list(new_idx)


def train_linear_torch_non_neg(train_features, train_targets, discrete=False, epochs=250):
    
    import torch
    if discrete:
        #print(int(max(train_targets)))
        if max(train_targets)==1:
            m = torch.nn.Linear(train_features.shape[1], int(max(train_targets)))
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            m = torch.nn.Linear(train_features.shape[1], 1+ int(max(train_targets)))
            criterion = torch.nn.CrossEntropyLoss()
    else:
        m = torch.nn.Linear(train_features.shape[1], train_targets.shape[1])
        criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(m.parameters(), lr = 1e-3)
    all_losses = []
    batch = 2048
    N = train_features.shape[0]//batch + 1
    for e in range(epochs):
        m = m.to(device)
        m.train()
        e_loss = 0
        for i in range(N):
            batch_features = torch.Tensor(train_features[i*batch:(i+1)*batch]).to(device) 
            batch_targets = torch.Tensor(train_targets[i*batch:(i+1)*batch]).to(device) 
            if discrete and int(max(train_targets))>1:
                batch_targets = batch_targets.to(torch.long)
            
            sc = m(batch_features)

            loss = criterion(sc, batch_targets)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            for p in m.parameters():
                p.data.clamp_(0)

            e_loss+=loss
        all_losses.append(e_loss.detach().cpu().numpy())
    
    sc_final = m(torch.Tensor(train_features).to(device))
    final_loss = criterion(sc_final, torch.Tensor(train_targets).to(device))
    all_losses.append(final_loss)
    return {'model':m.state_dict(), 'optim':optimizer.state_dict(), 'losses':all_losses}

def train_linear_torch(train_features, train_targets, discrete=False, epochs=100, print_out=False):
    
    import torch
    if discrete:
        #print(int(max(train_targets)))
        if max(train_targets)==1:
            m = torch.nn.Linear(train_features.shape[1], int(max(train_targets)))
            criterion = torch.nn.BCEWithLogitsLoss()
        else:
            m = torch.nn.Linear(train_features.shape[1], 1+ int(max(train_targets)))
            criterion = torch.nn.CrossEntropyLoss()
    else:
        m = torch.nn.Linear(train_features.shape[1], train_targets.shape[1])
        criterion = torch.nn.MSELoss()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    optimizer = torch.optim.Adam(m.parameters(), lr = 1e-3)
    all_losses = []
    batch = 2048
    N = train_features.shape[0]//batch + 1
    for e in range(epochs):
        m = m.to(device)
        m.train()
        e_loss = 0
        for i in range(N):
            batch_features = torch.Tensor(train_features[i*batch:(i+1)*batch]).to(device) 
            batch_targets = torch.Tensor(train_targets[i*batch:(i+1)*batch]).to(device) 
            if discrete and int(max(train_targets))>1:
                batch_targets = batch_targets.to(torch.long)
            
            sc = m(batch_features)

            loss = criterion(sc, batch_targets)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            e_loss+=loss
        all_losses.append(e_loss.detach().cpu().numpy())
        if print_out:
            print(e, e_loss, flush=True)
    return {'model':m.state_dict(), 'optim':optimizer.state_dict(), 'losses':all_losses}


def train_linear_block_torch(train_features, train_targets, num_attr, discrete=False, epochs=10):
    
    import torch
    import model
    
    print(time.time())

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    if discrete:
        #print(int(max(train_targets)))
        if max(train_targets)==1:
            m = model.block_linear(train_features.shape[1]//num_attr, 1, num_attr, device)
            criterion = torch.nn.BCEWithLogitsLoss()
            outsize=1
        else:
            m = model.block_linear(train_features.shape[1]//num_attr, int(max(train_targets)+1), num_attr, device)
            criterion = torch.nn.CrossEntropyLoss()
            outsize = int(max(train_targets)+1)
    else:
        m = model.block_linear(train_features.shape[1]//num_attr, train_targets.shape[1], num_attr, device)
        criterion = torch.nn.MSELoss()
        outsize = train_targets.shape[1]


    optimizer = torch.optim.Adam(m.parameters(), lr = 1e-4)
    all_losses = []
    batch = 1024
    N = train_features.shape[0]//batch + 1
    for e in range(epochs):
        m = m.to(device)
        m.train()
        print(num_attr, train_features.shape)
        e_loss = torch.zeros(num_attr)
        for i in range(N):
            batch_features = torch.Tensor(train_features[i*batch:(i+1)*batch]).to(device) 
            batch_targets = torch.Tensor(train_targets[i*batch:(i+1)*batch]).to(device) 
            if discrete and int(max(train_targets))>1:
                batch_targets = batch_targets.to(torch.long)
            
            sc = m(batch_features)
            
            loss_per_attr = [criterion(sc[:, a*outsize:(a+1)*outsize], batch_targets) for a in range(num_attr)]
            #print(loss_per_attr)
            loss = sum(loss_per_attr)
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            
            e_loss+=torch.Tensor(loss_per_attr)
        print(e, i, loss, time.time(), flush=True)
        all_losses.append(e_loss.detach().cpu().numpy())
    return {'model':m.state_dict(), 'optim':optimizer.state_dict(), 'losses':all_losses}
