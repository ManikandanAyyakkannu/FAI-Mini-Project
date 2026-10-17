# AI-Based University Enquiry Chatbot (Knowledge Representation + NLP)

M.Tech Fundamentals of AI, mini project, topic 29.
Team: Manikandan Ayyakkannu (RA2612049015035), Dinakar B (RA2612049015032), Manoj Kumar R (RA2612049015005).

**Try it online: https://manikandanayyakkannu.github.io/FAI-Mini-Project/**

This is a complete enquiry chatbot for the fictional *AI University*: admissions and programmes, fees, the academic calendar, exams and results, certificates, university offices, departments, hostel, placements, events, research and alumni. It runs completely in the browser, with no server, no API key and nothing to install. It combines:
- **frame-based knowledge representation** with `is_a` inheritance (a semantic network)
- **forward-chaining production rules** for eligibility and scholarship decisions, with explanations
- an **NLP pipeline** (normalisation, spelling correction, synonyms, stemming, entity extraction, intent classification)
- a **dialogue manager** that handles follow-up questions and asks for missing details
- a **vector store** fallback for free-text FAQs
- the **closed-world assumption**: a department the university does not have (e.g. Civil, Law) or a question about another university (e.g. SRM) gets a polite "not available" reply instead of a guess

There is no large language model inside. Every answer comes from a frame, a rule or an FAQ passage; the notebook shows the full trace for each answer.

## Project layout
| Path | What it is |
|---|---|
| `index.html` | The chatbot page (this is what opens online) |
| `web/css/style.css` | Page design (light and dark theme, works on phones) |
| `web/js/` | The chatbot in JavaScript, one file per part: `knowledge-base.js`, `nlp.js`, `intent-classifier.js`, `reasoning.js`, `answers.js`, `similarity.js`, `chatbot.js` (dialogue manager), `app.js` (the screen) |
| `data/` | Knowledge base, rules, intents, FAQs and test sets. **Edit these to change what the bot knows.** |
| `tests/engine-test.html` | Runs all four test sets in the browser and shows the accuracy |
| `University_Enquiry_Chatbot.ipynb` | Solution notebook (all 9 requirements, with outputs) |
| `University_Enquiry_Chatbot_Report.docx` / `.pdf` | Project report |
| `chatbot/` | The same engine in Python, used by the notebook and for the evaluation |
| `tools/` | Scripts that regenerate the report and the notebook |

The knowledge base has 70 frames: 20 programmes (engineering, arts, science, commerce, management), 11 departments, 9 university offices, 7 kinds of certificate, the academic calendar, exams, fee payment, events, research and alumni.
The JavaScript and Python versions give identical answers on all 361 test conversations.

## Using it
- **Chat:** open the link above. Click a sample question on the left or type your own.
- **Presentation:** click *Demo 1*, *Demo 2* or *Demo 3* in the left panel, or open the link with `?demo=1`, `?demo=2` or `?demo=3` added at the end.
- **Test results:** open `tests/engine-test.html` on the same site.

## Putting it online (GitHub Pages, one time)
1. Push this folder to the GitHub repository `FAI-Mini-Project`.
2. On GitHub open **Settings > Pages**, set *Source* to **Deploy from a branch**, branch **main**, folder **/ (root)**, and click **Save**.
3. After about a minute the site is live at the link above.

Opening `index.html` by double-clicking it does not work, because browsers block a page opened from a folder from reading the `data/` files. Use the online link, or the *Live Server* extension in VS Code.

## Notebook (optional, for the evaluation)
Open the folder in VS Code, open `University_Enquiry_Chatbot.ipynb`, choose the **Python 3.12** kernel and click **Run All**. The libraries are listed in `requirements.txt`.

## Results (share of correct answers)
| Test set | This chatbot | NLP + vector retrieval | Keyword FAQ matching |
|---|---|---|---|
| Admissions - main (105) | **94.3%** (intent macro-F1 0.96) | 32.4% | 26.7% |
| Admissions - held-out (25) | **80.0%** | 48.0% | 32.0% |
| University services - development (40) | **100%** | 45.0% | 42.5% |
| University services - held-out (20) | **90.0%** | 40.0% | 40.0% |

The held-out sets were not used to build or tune the chatbot. The university-services held-out set scored 80.0% when first run; it exposed a spelling-correction bug ("using" was corrected to "nursing"), and after that fix it scores 90.0%. The development set was used while building the university-services features, so its 100% is optimistic.
