/**
 * Le righe di codice Python VERE che le azioni evidenziano nel pannello del codice (SOLI DATI). Ogni
 * voce è [chiave dello snippet di data/model.json, testo della riga così com'è nel file]: un test
 * (tests/lab-actions.test.mjs) controlla che ogni riga esista davvero nel suo snippet, così una riga
 * cambiata in gpt/*.py non resta citata a vuoto.
 */
export const CODE = Object.freeze({
  key: ["head", "k = self.key(x)"],
  query: ["head", "q = self.query(x)"],
  value: ["head", "v = self.value(x)"],
  scores: ["head", "wei = q @ k.transpose(-2, -1) * self.head_size**-0.5"],
  mask: ["head", 'wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))'],
  softmax: ["head", "return F.softmax(wei, dim=-1)"],
  weightedSum: ["head", "return wei @ v"],
  concat: ["mha", "out = torch.cat([h(x) for h in self.heads], dim=-1)"],
  proj: ["mha", "return self.dropout(self.proj(out))"],
  residualAttention: ["block", "x = x + self.sa(self.ln1(x))"],
  residualFfn: ["block", "x = x + self.ffwd(self.ln2(x))"],
  ffnForward: ["ffwd", "return self.net(x)"],
  ffn1: ["ffwd", "nn.Linear(config.n_embd, hidden),"],
  relu: ["ffwd", "nn.ReLU(),"],
  ffn2: ["ffwd", "nn.Linear(hidden, config.n_embd),"],
  lnF: ["model.forward", "x = self.ln_f(x)"],
  lmHead: ["model.forward", "logits = self.lm_head(x)"],
  probs: ["generate", "probs = F.softmax(logits, dim=-1)"],
  sample: ["generate", "idx_next = torch.multinomial(probs, num_samples=1, generator=generator)"],
  loss: ["loss", "return F.cross_entropy(logits.view(B * T, V), targets.view(B * T))"],
  forwardLoss: ["train", "_, loss = model(xb, yb)"],
  backward: ["train", "loss.backward()"],
  step: ["train", "optimizer.step()"],
  adamw: ["train", "optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)"],
  loop: ["train", "for step in range(train_cfg.max_iters):"],
});

/**
 * Gli argomenti di stage.code() per una voce: lo snippet, un titolo (la riga e dove siamo) e la riga
 * da evidenziare. `where` = "head 0 del blocco 0", "lm_head"…
 */
export function codeFor(id, where = "") {
  const [snippet, line] = CODE[id];
  const title = line.replace(/,\s*$/, ""); // "nn.ReLU()," è un elemento di nn.Sequential: nel titolo senza virgola
  return [snippet, where ? `${title}  ·  ${where}` : title, [line]];
}
