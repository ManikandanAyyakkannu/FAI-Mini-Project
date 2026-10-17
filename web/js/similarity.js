// Text similarity helpers used by the chatbot:
//   1. closestWord()  - spelling correction (same scoring as Python's difflib)
//   2. TfidfEmbedder  - turns text into a vector (word + character n-grams)
//   3. VectorStore    - stores vectors and finds the most similar ones

// ---------------------------------------------------------------- 1. spelling
// Counts the characters two words have in common, matching blocks the way
// difflib.SequenceMatcher does (longest common block first, then left and right).
function matchingCharacters(a, b) {
  const positions = new Map();          // character of b -> list of its indexes
  for (let j = 0; j < b.length; j++) {
    if (!positions.has(b[j])) positions.set(b[j], []);
    positions.get(b[j]).push(j);
  }

  function longestBlock(alo, ahi, blo, bhi) {
    let bestI = alo, bestJ = blo, bestSize = 0;
    let lengths = new Map();
    for (let i = alo; i < ahi; i++) {
      const newLengths = new Map();
      for (const j of positions.get(a[i]) || []) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (lengths.get(j - 1) || 0) + 1;
        newLengths.set(j, k);
        if (k > bestSize) { bestI = i - k + 1; bestJ = j - k + 1; bestSize = k; }
      }
      lengths = newLengths;
    }
    return [bestI, bestJ, bestSize];
  }

  let total = 0;
  const todo = [[0, a.length, 0, b.length]];
  while (todo.length) {
    const [alo, ahi, blo, bhi] = todo.pop();
    const [i, j, k] = longestBlock(alo, ahi, blo, bhi);
    if (k) {
      total += k;
      if (alo < i && blo < j) todo.push([alo, i, blo, j]);
      if (i + k < ahi && j + k < bhi) todo.push([i + k, ahi, j + k, bhi]);
    }
  }
  return total;
}

function commonLetters(a, b) {
  const count = {};
  for (const c of b) count[c] = (count[c] || 0) + 1;
  let n = 0;
  for (const c of a) if (count[c] > 0) { count[c]--; n++; }
  return n;
}

// Returns the most similar candidate whose similarity is at least `cutoff`, or null.
// Among equally close candidates a word from `preferred` wins ("fess" -> "fees", not "fest").
export function closestWord(word, candidates, cutoff, preferred = new Set()) {
  const found = [];
  for (const cand of candidates) {
    const total = cand.length + word.length;
    // two cheap upper bounds first, then the real score
    if (2 * Math.min(cand.length, word.length) / total < cutoff) continue;
    if (2 * commonLetters(cand, word) / total < cutoff) continue;
    const score = 2 * matchingCharacters(cand, word) / total;
    if (score >= cutoff) found.push([score, cand]);
  }
  if (!found.length) return null;
  // the 10 closest (like difflib.get_close_matches), then the preferred word on a tie
  found.sort((a, b) => b[0] - a[0] || (a[1] < b[1] ? 1 : a[1] > b[1] ? -1 : 0));
  let best = null;
  for (const [score, cand] of found.slice(0, 10)) {
    const key = [score, preferred.has(cand) ? 1 : 0, cand];
    if (best === null || key[0] > best[0] || (key[0] === best[0] && (key[1] > best[1] ||
        (key[1] === best[1] && key[2] > best[2])))) best = key;
  }
  return best[2];
}

// ---------------------------------------------------------------- 2. TF-IDF
// Same settings as scikit-learn's TfidfVectorizer(sublinear_tf=True):
// idf = ln((1 + n) / (1 + df)) + 1, tf = 1 + ln(count), each vector has length 1.
class TfidfVectorizer {
  constructor(analyzer, minN, maxN) {
    this.analyzer = analyzer;
    this.minN = minN;
    this.maxN = maxN;
    this.idf = new Map();
  }

  terms(text) {
    text = text.toLowerCase();
    const out = [];
    if (this.analyzer === "word") {
      const tokens = text.match(/\b\w\w+\b/g) || [];
      for (let n = this.minN; n <= this.maxN; n++)
        for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
      return out;
    }
    // "char_wb": character n-grams inside each word, padded with a space on both sides
    for (const raw of text.split(/\s+/)) {
      if (!raw) continue;
      const w = " " + raw + " ";
      for (let n = this.minN; n <= this.maxN; n++) {
        if (w.length <= n) { out.push(w); break; }   // a short word is counted only once
        for (let i = 0; i + n <= w.length; i++) out.push(w.slice(i, i + n));
      }
    }
    return out;
  }

  fit(texts) {
    const df = new Map();
    for (const t of texts)
      for (const term of new Set(this.terms(t))) df.set(term, (df.get(term) || 0) + 1);
    const n = texts.length;
    this.idf = new Map([...df].map(([term, d]) => [term, Math.log((1 + n) / (1 + d)) + 1]));
    return this;
  }

  transform(text) {
    const counts = new Map();
    for (const term of this.terms(text))
      if (this.idf.has(term)) counts.set(term, (counts.get(term) || 0) + 1);
    const vec = new Map();
    let norm = 0;
    for (const [term, c] of counts) {
      const w = (1 + Math.log(c)) * this.idf.get(term);
      vec.set(term, w);
      norm += w * w;
    }
    norm = Math.sqrt(norm);
    if (norm > 0) for (const [term, w] of vec) vec.set(term, w / norm);
    return vec;
  }
}

// Word 1-2 grams plus character 3-5 grams (weight 0.7). Character n-grams make the
// match tolerant of typing mistakes.
export class TfidfEmbedder {
  constructor() {
    this.word = new TfidfVectorizer("word", 1, 2);
    this.char = new TfidfVectorizer("char_wb", 3, 5);
  }

  fit(texts) {
    this.word.fit(texts);
    this.char.fit(texts);
    return this;
  }

  embed(text) {
    const vec = new Map();
    for (const [t, w] of this.word.transform(text)) vec.set("w:" + t, w);
    for (const [t, w] of this.char.transform(text)) vec.set("c:" + t, w * 0.7);
    let norm = 0;
    for (const w of vec.values()) norm += w * w;
    norm = Math.sqrt(norm);
    if (norm > 0) for (const [t, w] of vec) vec.set(t, w / norm);
    return vec;
  }
}

// cosine similarity of two vectors that already have length 1
export function dot(a, b) {
  if (a.size > b.size) [a, b] = [b, a];
  let s = 0;
  for (const [t, w] of a) {
    const v = b.get(t);
    if (v !== undefined) s += w * v;
  }
  return s;
}

// ---------------------------------------------------------------- 3. store
export class VectorStore {
  constructor() {
    this.embedder = new TfidfEmbedder();
    this.texts = [];
    this.metadata = [];
    this.vectors = [];
  }

  add(texts, metadata) {
    this.texts.push(...texts);
    this.metadata.push(...metadata);
    this.embedder.fit(this.texts);        // re-fit so the vocabulary covers every document
    this.vectors = this.texts.map(t => this.embedder.embed(t));
  }

  search(query, k = 3) {
    const q = this.embedder.embed(query);
    return this.vectors
      .map((v, i) => ({ score: dot(v, q), text: this.texts[i], metadata: this.metadata[i] }))
      .sort((x, y) => y.score - x.score)
      .slice(0, k);
  }

  get size() {
    return this.texts.length;
  }
}
