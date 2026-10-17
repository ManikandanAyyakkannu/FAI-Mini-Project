// Hybrid intent classifier.
//
//   score(intent) = 0.6 x similarity + 0.4 x keyword evidence
//
// similarity = highest cosine similarity between the question and the intent's
//              example sentences (TF-IDF vectors of the pre-processed text)
// keyword    = 1 if a trigger word of the intent appears in the question, else 0
//
// The best-scoring intent wins. If even the best score is below the threshold,
// the question is treated as out of scope.

import { TfidfEmbedder, dot } from "./similarity.js";
import { TextPreprocessor } from "./nlp.js";

const W_SIM = 0.6, W_KW = 0.4;

export class IntentClassifier {
  constructor(pre, data, threshold = 0.30) {
    this.intents = data.intents;
    this.smalltalk = data.smalltalk_responses;
    this.pre = pre;
    this.threshold = threshold;

    // the words of the example sentences also feed the spelling corrector
    for (const spec of Object.values(this.intents))
      for (const ex of spec.examples) pre.addVocabulary(TextPreprocessor.normalise(ex).split(" "));

    this.labels = [];
    const texts = [];
    for (const [name, spec] of Object.entries(this.intents)) {
      for (const ex of spec.examples) {
        this.labels.push(name);
        texts.push(pre.process(ex).lemmaText || ex);
      }
    }
    this.embedder = new TfidfEmbedder().fit(texts);
    this.vectors = texts.map(t => this.embedder.embed(t));
    this.keywords = {};
    for (const [name, spec] of Object.entries(this.intents)) {
      this.keywords[name] = new Set(spec.keywords.map(k => pre.process(k).lemmaText).filter(Boolean));
    }
  }

  scores(processed) {
    const out = {};
    const text = processed.lemmaText;
    if (!text) {
      for (const name in this.intents) out[name] = 0;
      return out;
    }
    const q = this.embedder.embed(text);
    const sims = this.vectors.map(v => dot(v, q));
    const lemmaSet = new Set(processed.lemmas);
    const joined = ` ${text} `;
    for (const name in this.intents) {
      let sim = -Infinity;
      this.labels.forEach((label, i) => { if (label === name && sims[i] > sim) sim = sims[i]; });
      const kw = [...this.keywords[name]].some(k => (k.includes(" ") ? joined.includes(` ${k} `) : lemmaSet.has(k)));
      out[name] = W_SIM * sim + W_KW * (kw ? 1 : 0);
    }
    return out;
  }

  predict(processed) {
    const sc = this.scores(processed);
    let best = null;
    for (const name in sc) if (best === null || sc[name] > sc[best]) best = name;
    const intent = sc[best] < this.threshold ? "out_of_scope" : best;
    return { intent, confidence: sc[best], scores: sc };
  }
}
