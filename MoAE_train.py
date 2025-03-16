import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from MoE import MoAE  # Ensure MoE module is available


# =========================== #
#      Distributed Setup      #
# =========================== #

def setup(rank, world_size):
    """Initialize the distributed training environment."""
    dist.init_process_group(
        backend="nccl",  # Use "gloo" if running on CPU
        init_method="tcp://127.0.0.1:29500",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


# =========================== #
#     Model Definition        #
# =========================== #

class MoAETransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, n_head_moe, experts, top_n, dropout, is_distributed, batch_first):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head, batch_first=batch_first)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.moae = MoAE(
            dim=d_model,
            num_experts=experts,
            num_heads=n_head_moe,
            threshold_train=0.2,
            threshold_eval=0.2,
            capacity_factor_train=1.25,
            capacity_factor_eval=2.,
            gating_top_n=top_n,
            balance_loss_coef=1e-2,
            router_z_loss_coef=1e-3,
            straight_through_dispatch_tensor=True,
            differentiable_topk=False,
            differentiable_topk_fused=True,
            is_distributed=is_distributed
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, key_padding_mask=None, attn_mask=None, moe_is_causal=False):
        attn_out, _ = self.attention(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        moe_out, total_aux_loss, _, _ = self.moae(x, key_padding_mask, moe_is_causal)
        x = self.norm2(x + self.dropout(moe_out))
        return x, total_aux_loss


class MoAETransformer(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, n_head, n_head_moe, experts, top_n, n_layers, dropout, is_distributed, batch_first):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.transformer_layers = nn.ModuleList([
            MoAETransformerBlock(d_model, n_head, n_head_moe, experts, top_n, dropout, is_distributed, batch_first)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size)

    def get_causal_mask(self, seq_length):
        return torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool()

    def forward(self, x, key_padding_mask=None, is_causal=False):
        B, S = x.shape
        x = self.embedding(x)
        pos_embed = self.pos_embedding(torch.arange(S, device=x.device))
        x = x + pos_embed
        attn_mask = self.get_causal_mask(S).to(x.device)

        total_aux_loss = torch.tensor(0.0, device=x.device)
        for layer in self.transformer_layers:
            x, aux_loss = layer(x, key_padding_mask, attn_mask, is_causal)
            total_aux_loss += aux_loss.squeeze()

        return self.head(x), total_aux_loss

    def generate(self, idx, max_new_tokens=128):
        """Generates text using the trained model."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =========================== #
#     Training Functions      #
# =========================== #

def get_batch(split, train_data, val_data, block_size, batch_size, device):
    """Generate a batch of data for training or validation."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(torch.long)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(torch.long)
    return x.to(device), y.to(device)


@torch.no_grad()
def evaluate(model, val_data, loss_fn, config):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0.0
    for _ in range(config["eval_iters"]):
        X, Y = get_batch('val', config["train_data"], val_data, config["block_size"], config["batch_size"], config["device"])
        logits, aux_loss = model(X, is_causal=True)
        loss = loss_fn(logits.view(-1, config["vocab_size"]), Y.view(-1)) + aux_loss
        total_loss += loss.item()
    
    avg_loss = total_loss / config["eval_iters"]
    model.train()
    return avg_loss

def build_vocab(text):
    """Builds vocabulary mappings from characters to indices and vice versa."""
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos

def encode(text, stoi):
    """Encodes a string into a list of integers using the given character-to-index mapping."""
    return [stoi[c] for c in text]

def decode(indices, itos):
    """Decodes a list of integers into a string using the given index-to-character mapping."""
    return ''.join([itos[i] for i in indices])

def print_sample(model,context,config):
    decode_fn = config['decode_fn']
    itos = config['itos']
    model.eval()            
    text = decode_fn(model.module.generate(context, max_new_tokens=500)[0].tolist(),itos)
    model.train()
    return text

def run(rank, world_size, config):
    """Main training and evaluation loop."""
    setup(rank, world_size)

    # Initialize model
    model = MoAETransformer(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        max_len=config["block_size"],
        n_head=config["n_head"],
        n_head_moe=config["n_head_moe"],
        experts=config["experts"],
        top_n=config["top_n"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        is_distributed=True,
        batch_first=True
    ).to(rank)

    model = DDP(model, device_ids=[rank],find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    for iter in range(config["max_iters"]):
        X, Y = get_batch('train', config["train_data"], config["val_data"], config["block_size"], config["batch_size"], rank)
        outputs, aux_loss = model(X, is_causal=True)
        loss = loss_fn(outputs.view(-1, config["vocab_size"]), Y.view(-1)) + aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        if (iter % config["eval_interval"] == 0 or iter == config["max_iters"] - 1):
            avg_loss = evaluate(model, config["val_data"], loss_fn, config)
            context = torch.zeros((1, 1), dtype=torch.long, device=rank)
            text = print_sample(model,context,config)
            if rank == 0:
                print(f"Rank {rank} - Step {iter}")
                print(f"Training Loss: {loss.item():.4f}")
                print(f"Validation loss {avg_loss:.4f}")
                print(f"Generated text:", text)


        dist.barrier()
                

    cleanup()


# =========================== #
#        Main Execution       #
# =========================== #

if __name__ == "__main__":
    # Configurable hyperparameters
    config = {
        "batch_size": 256,
        "block_size": 32,
        "max_iters": 2500,
        "eval_interval": 100,
        "learning_rate": 3e-4,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "eval_iters": 200,
        "d_model": 128,
        "n_head": 4,
        "n_head_moe": 4,
        "experts": 4,
        "top_n": 2,
        "n_layers": 4,
        "dropout": 0.0,
    }

    # Load dataset
    with open('shakespear.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    # create a mapping from characters to integers
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }

    # Train and test splits
    data = torch.tensor(encode(text,stoi), dtype=torch.long)
    n = int(0.9*len(data)) # first 90% will be train, rest val
    train_data = data[:n]
    val_data = data[n:]

    config['vocab_size'] = len(chars)
    config['encode_fn'] = encode 
    config['decode_fn'] = decode
    config['train_data'] = train_data
    config['val_data'] = val_data
    config['stoi'] = stoi
    config['itos'] = itos

    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size, config,), nprocs=world_size, join=True)
