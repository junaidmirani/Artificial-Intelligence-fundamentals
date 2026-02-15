# tiny_cnn_algorithm_art.py
# Pure pedagogical, "algorithm-art" single-file tiny CNN implemented with scalar Value autograd.
# Style and spirit inspired by Karpathy: explicit loops, tiny, readable, educational.
#
# Run: python tiny_cnn_algorithm_art.py

import math, random
random.seed(42)


# Autograd: scalar Value
class Value:
    __slots__ = ('data','grad','_children','_local_grads')
    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = tuple(children)
        self._local_grads = tuple(local_grads)

    # basic ops (create new Value nodes and record local grads)
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))
    def __radd__(self, other): return self + other
    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return self * (other ** -1)
    def __rtruediv__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return other * (self ** -1)
    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other-1),))

    def relu(self):
        return Value(self.data if self.data > 0 else 0.0, (self,), (1.0 if self.data > 0 else 0.0,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    # reverse-mode autodiff
    def backward(self):
        topo = []
        visited = set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, local in zip(v._children, v._local_grads):
                child.grad += local * v.grad


# tiny linear algebra helpers (list-of-Value)
def zeros_matrix(nout, nin): return [[Value(0.0) for _ in range(nin)] for _ in range(nout)]
def rand_matrix(nout, nin, std=0.1): return [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

def linear(x, w, bias=None):
    # w: list of rows; x: list of Value
    out = []
    for i, row in enumerate(w):
        s = Value(0.0)
        for wi, xi in zip(row, x):
            s = s + wi * xi
        if bias is not None:
            s = s + bias[i]
        out.append(s)
    return out

def softmax(logits):
    # keep numerically stable, but remain fully Value-based for grads
    mx = max(l.data for l in logits)
    exps = [(l - mx).exp() for l in logits]
    tot = sum(exps)
    return [e / tot for e in exps]


# Conv2D (valid), average pool 2x2, flatten
# All implemented with explicit loops and Value ops.
def conv2d(input_ch_hw, kernels, biases):
    # input_ch_hw: [in_ch][H][W] of Value
    # kernels: [out_ch][in_ch][kh][kw] of Value
    in_ch = len(input_ch_hw)
    H = len(input_ch_hw[0])
    W = len(input_ch_hw[0][0])
    kh = len(kernels[0][0])
    kw = len(kernels[0][0][0])
    out_h = H - kh + 1
    out_w = W - kw + 1
    out = []
    for fo, filt in enumerate(kernels):
        fmap = [[None for _ in range(out_w)] for _ in range(out_h)]
        for i in range(out_h):
            for j in range(out_w):
                s = Value(0.0)
                for c in range(in_ch):
                    for yi in range(kh):
                        for xi in range(kw):
                            s = s + filt[c][yi][xi] * input_ch_hw[c][i+yi][j+xi]
                s = s + biases[fo]
                fmap[i][j] = s
        out.append(fmap)
    return out  # [out_ch][out_h][out_w]

def avg_pool2x2(fmaps):
    out = []
    for fmap in fmaps:
        H = len(fmap); W = len(fmap[0])
        assert H % 2 == 0 and W % 2 == 0
        ph, pw = H//2, W//2
        pooled = [[None for _ in range(pw)] for _ in range(ph)]
        for i in range(ph):
            for j in range(pw):
                s = Value(0.0)
                s = s + fmap[2*i][2*j]
                s = s + fmap[2*i][2*j+1]
                s = s + fmap[2*i+1][2*j]
                s = s + fmap[2*i+1][2*j+1]
                pooled[i][j] = s / 4.0
        out.append(pooled)
    return out

def flatten(fmaps):
    flat = []
    for c in fmaps:
        for row in c:
            for v in row:
                flat.append(v)
    return flat


# tiny synthetic dataset: 8x8 single-channel images
# classes: vertical-bar vs horizontal-bar
# purely procedural, pedagogical, and minimal
H = W = 8
def make_bar(kind, offx=0, offy=0, width=3):
    img = [[0.0 for _ in range(W)] for _ in range(H)]
    if kind == 'vertical':
        cx = 3 + offx
        for i in range(H):
            for j in range(max(0,cx-width//2), min(W, cx+width//2+1)):
                img[i][j] = 1.0
    else:
        cy = 3 + offy
        for i in range(max(0,cy-width//2), min(H, cy+width//2+1)):
            for j in range(W):
                img[i][j] = 1.0
    return img

# small dataset (deterministic)
dataset = []
labels = []
for off in range(-1,2):   # small positional variation for some challenge
    dataset.append(make_bar('vertical', offx=off))
    labels.append(0)
    dataset.append(make_bar('horizontal', offy=off))
    labels.append(1)

# shuffle
pairs = list(zip(dataset, labels))
random.shuffle(pairs)
dataset, labels = zip(*pairs)
dataset, labels = list(dataset), list(labels)


# model: 1 conv layer, avg pool, linear head
# explicit and tiny
in_ch = 1
n_filters = 4
k = 3  # 3x3 kernels

kernels = [
    [ [ [Value(random.gauss(0, 0.1)) for _ in range(k)] for _ in range(k) ] for __ in range(in_ch) ]
    for _ in range(n_filters)
]
biases = [Value(0.0) for _ in range(n_filters)]

post_h = (H - k + 1) // 2
post_w = (W - k + 1) // 2
flat_len = n_filters * post_h * post_w

head_w = rand_matrix(2, flat_len, std=0.2)
head_b = [Value(0.0) for _ in range(2)]

# collect params
params = []
for filt in kernels:
    for ch in filt:
        for row in ch:
            for p in row:
                params.append(p)
for b in biases: params.append(b)
for row in head_w:
    for p in row: params.append(p)
for b in head_b: params.append(b)

print(f"num params: {len(params)}")


# forward (wrap numeric img into Value grid)
def to_values(img):
    return [[ [Value(img[i][j]) for j in range(W)] for i in range(H) ]]

def forward(img):
    x = to_values(img)  # [ch=1][H][W]
    conv = conv2d(x, kernels, biases)
    # relu
    for f in range(len(conv)):
        for i in range(len(conv[f])):
            for j in range(len(conv[f][i])):
                conv[f][i][j] = conv[f][i][j].relu()
    pooled = avg_pool2x2(conv)
    flat = flatten(pooled)
    logits = linear(flat, head_w, bias=head_b)
    return logits

def ce_loss(logits, label):
    probs = softmax(logits)
    return -probs[label].log()


# optimizer: simple Adam (float state)
lr = 0.01
b1, b2 = 0.9, 0.99
eps = 1e-8
m = [0.0]*len(params)
v = [0.0]*len(params)


# training loop: single-sample SGD for clarity
steps = 300
for step in range(steps):
    i = step % len(dataset)
    img = dataset[i]
    lbl = labels[i]

    logits = forward(img)
    loss = ce_loss(logits, lbl)
    loss.backward()

    # adam update
    for idx, p in enumerate(params):
        g = p.grad
        m[idx] = b1*m[idx] + (1-b1)*g
        v[idx] = b2*v[idx] + (1-b2)*(g*g)
        mhat = m[idx] / (1 - b1**(step+1))
        vhat = v[idx] / (1 - b2**(step+1))
        p.data -= lr * (mhat / (math.sqrt(vhat) + eps))
        p.grad = 0.0

    if (step+1) % 50 == 0 or step == 0:
        print(f"step {step+1:4d}/{steps:4d} | loss {loss.data:.6f}")


# inference: sample a few generations (class predictions)
print("\n--- evaluation on dataset ---")
correct = 0
for img, lbl in zip(dataset, labels):
    logits = forward(img)
    probs = softmax(logits)
    pred = 0 if probs[0].data > probs[1].data else 1
    if pred == lbl:
        correct += 1
print(f"accuracy: {correct}/{len(dataset)}")

# show learned kernels (as small matrices) — readable algorithm-art
print("\nlearned kernels (per-filter):")
for fi, filt in enumerate(kernels):
    print(f"filter {fi}: bias={biases[fi].data:.4f}")
    for row in filt[0]:
        print([round(p.data,4) for p in row])
    print("-"*32)

# visual check: a few perturbed inputs
print("\nrobustness to small shifts:")
def shift(img, dx, dy):
    out = [[0.0 for _ in range(W)] for _ in range(H)]
    for y in range(H):
        for x in range(W):
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W:
                out[ny][nx] = img[y][x]
    return out

for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(2,0),(0,2)]:
    ok = 0
    for img, lbl in zip(dataset, labels):
        img2 = shift(img, dx, dy)
        logits = forward(img2)
        probs = softmax(logits)
        pred = 0 if probs[0].data > probs[1].data else 1
        if pred == lbl: ok += 1
    print(f"shift ({dx:2d},{dy:2d}) acc: {ok}/{len(dataset)}")
