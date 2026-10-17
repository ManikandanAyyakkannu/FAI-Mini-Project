# AI-Based University Enquiry Chatbot (Knowledge Representation + NLP)

M.Tech Fundamentals of AI, mini project, topic 29.
Team: Manikandan (Roll No. 35), Manoj Kumar (Roll No. 05), Dinakar (Roll No. 32).

**Try it online: https://manikandanayyakkannu.github.io/FAI-Mini-Project/**

This is an enquiry chatbot for the fictional *Greenfield University*. It runs completely in the browser, with no server, no API key and nothing to install. It combines:
- **frame-based knowledge representation** with `is_a` inheritance (a semantic network)
- **forward-chaining production rules** for eligibility and scholarship decisions, with explanations
- an **NLP pipeline** (normalisation, spelling correction, synonyms, stemming, entity extraction, intent classification)
- a **dialogue manager** that handles follow-up questions and asks for missing details
- a **vector store** fallback for free-text FAQs

There is no large language model inside. Every answer comes from a frame or a rule, and the "How I answered" panel shows which one.

## Project layout
| Path | What it is |
|---|---|
| `index.html` | The chatbot page (this is what opens online) |
| `web/css/style.css` | Page design (light and dark theme, works on phones) |
| `web/js/` | The chatbot in JavaScript, one file per part: `knowledge-base.js`, `nlp.js`, `intent-classifier.js`, `reasoning.js`, `answers.js`, `similarity.js`, `chatbot.js` (dialogue manager), `app.js` (the screen) |
| `data/` | Knowledge base, rules, intents, FAQs and test sets. **Edit these to change what the bot knows.** |
| `tests/engine-test.html` | Runs both test sets in the browser and shows the accuracy |
| `University_Enquiry_Chatbot.ipynb` | Solution notebook (all 9 requirements, with outputs) |
| `University_Enquiry_Chatbot_Report.docx` / `.pdf` | Project report |
| `chatbot/` | The same engine in Python, used by the notebook and for the evaluation |
| `tools/` | Scripts that regenerate the report and the notebook |

The JavaScript and Python versions give identical answers on all 154 test conversations.

## Using it
- **Chat:** open the link above. Click a sample question on the left or type your own.
- **Presentation:** click *Demo 1* or *Demo 2* in the left panel, or open the link with `?demo=1` / `?demo=2` added at the end.
- **Test results:** open `tests/engine-test.html` on the same site.

## Putting it online (GitHub Pages, one time)
1. Push this folder to the GitHub repository `FAI-Mini-Project`.
2. On GitHub open **Settings > Pages**, set *Source* to **Deploy from a branch**, branch **main**, folder **/ (root)**, and click **Save**.
3. After about a minute the site is live at the link above.

Opening `index.html` by double-clicking it does not work, because browsers block a page opened from a folder from reading the `data/` files. Use the online link, or the *Live Server* extension in VS Code.

## Notebook (optional, for the evaluation)
Open the folder in VS Code, open `University_Enquiry_Chatbot.ipynb`, choose the **Python 3.12** kernel and click **Run All**. The libraries are listed in `requirements.txt`.

## Results (105-query test set)
| System | Answer accuracy | Intent macro-F1 |
|---|---|---|
| Keyword FAQ matching | 31.4% | 0.39 |
| NLP + vector retrieval | 38.1% | 0.49 |
| **Hybrid KR + NLP (this project)** | **92.4%** | **0.93** |

On the held-out set, which was scored once after development, the hybrid chatbot got 80.0% of answers right, against 48.0% and 32.0% for the two baselines.
