import torch
import torch.nn as nn
from torch.nn import functional as F
from MoE import MoAE



class MoAETrasnformerBlock(nn.Module):
    def __init__(self,
                d_model,
                n_head,
                n_head_moe,
                experts,
                top_n,
                dropout=0.1,
                is_distributed = None,
                batch_first=True):
        super(MoAETrasnformerBlock,self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head,batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moae = MoAE(d_model,
                        num_experts = experts,
                        num_heads = n_head_moe,
                        threshold_train = 0.2,
                        threshold_eval = 0.2,
                        capacity_factor_train = 1.25,
                        capacity_factor_eval = 2.,
                        gating_top_n = top_n,
                        balance_loss_coef = 1e-2,
                        router_z_loss_coef = 1e-3,
                        straight_through_dispatch_tensor = True,
                        differentiable_topk = False,
                        differentiable_topk_fused = True,
                        is_distributed = is_distributed,) 
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,key_padding_mask=None,attn_mask=None, moe_is_causal=False):
        attention_out,_ = self.attention(x,x,x,key_padding_mask=key_padding_mask,attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attention_out))
        moe_output, total_aux_loss, balance_loss, router_z_loss = self.moae(x,key_padding_mask,moe_is_causal)
        out = self.norm2(x + self.dropout(moe_output))
        return out, total_aux_loss, balance_loss, router_z_loss
    
class MoAETrasnformer(nn.Module):
    def __init__(self,
                vocab_size,
                d_model,
                max_len,
                n_head,
                n_head_moe,
                experts,
                top_n,
                n_layers,
                dropout=0.1,
                is_distributed = None,
                batch_first=True
                 ):
        super(MoAETrasnformer,self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.n_head = n_head
        self.experts = experts
        self.top_n = top_n
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_first = batch_first
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_embedding = nn.Embedding(max_len,d_model)
        self.transformer_layers = nn.ModuleList([
            MoAETrasnformerBlock(d_model,n_head,n_head_moe,experts,top_n,dropout,is_distributed,batch_first) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model,vocab_size)
    def get_causal_mask(self,seq_length):
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()
        return mask
    def forward(self,x,key_padding_mask=None,is_causal=False):
        B,S = x.shape
        x = self.embedding(x)
        pos_embed = self.pos_embedding(torch.arange(S, device=device))
        x = x + pos_embed
        attn_mask = self.get_causal_mask(x.shape[1]).to(x.device)
        total_aux_loss = torch.tensor(0.).to(x.device)
        for layer in self.transformer_layers:
            x,aux_loss,_,_ =layer(x,key_padding_mask,attn_mask,is_causal)
            total_aux_loss = total_aux_loss + aux_loss
        x = self.head(x)
        return x,total_aux_loss
    def generate(self,idx,max_new_tokens=128):
        B,T = idx.shape
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.max_len:]
            logits, _ = self(idx_cond)
            logits = logits[:,-1,:]
            probs = F.softmax(logits,dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_next),dim=1)
        return idx


# data loading
def get_batch(split,train_data,val_data):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]).to(torch.long)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]).to(torch.long)
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model,train_data,val_data):
    out = {}
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split,train_data,val_data)
            B,S = X.shape
            logits,aux_loss = model(X, is_causal=True)
            loss = loss_fn(logits.view(B*S,-1),Y.view(B*S)) + aux_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

    
if __name__ == "__main__":
    # hyperparameters
    batch_size = 128 # how many independent sequences will we process in parallel?
    block_size = 32 # what is the maximum context length for predictions?
    max_iters = 2500
    eval_interval = 100
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 64
    n_head = 4
    n_head_moe = 2
    experts = 4
    top_n = 2
    n_layer = 4
    dropout = 0.0
    is_distributed = False
    # ------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1337)

    # wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
    with open('shakespear.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # here are all the unique characters that occur in this text
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    # Train and test splits
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]


    model = MoAETrasnformer(
        vocab_size=vocab_size,
        d_model=n_embd,
        max_len=block_size, 
        n_head=n_head,
        n_head_moe=n_head_moe,
        experts=experts,
        top_n=top_n,
        n_layers=n_layer,
        dropout=dropout,
        is_distributed=is_distributed,
        batch_first=True
    )
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
    model = model.to(device)
    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model,train_data,val_data)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train',train_data,val_data)
        b,s = xb.shape
        # evaluate the loss
        outputs,aux_loss = model(xb,is_causal=True)
        loss = loss_fn(outputs.view(b*s,-1),yb.view(b*s)) + aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
    import pdb;pdb.set_trace()