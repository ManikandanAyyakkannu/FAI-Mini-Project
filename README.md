# AI-Based University Enquiry Chatbot (Knowledge Representation + NLP)

M.Tech Fundamentals of AI, mini project, topic 29.
Team: Manikandan (Roll No. 35), Manoj Kumar (Roll No. 05), Dinakar (Roll No. 32).

This is an enquiry chatbot for the fictional *Greenfield University*. It combines:
- **frame-based knowledge representation** with `is_a` inheritance (a semantic network)
- **forward-chaining production rules** for eligibility and scholarship decisions, with explanations
- an **NLP pipeline** (normalisation, spelling correction, synonyms, stemming, entity extraction, intent classification)
- a **dialogue manager** that handles follow-up questions and asks for missing details
- a **vector store** fallback for free-text FAQs

## Deliverables
| File | What it is |
|---|---|
| `University_Enquiry_Chatbot.ipynb` | Solution notebook. It covers all 9 requirements, with outputs saved |
| `chat_cli.py` | Terminal chatbot (submission version) |
| `app_streamlit.py` | Web chatbot for the live demo |
| `University_Enquiry_Chatbot_Report.docx` / `.pdf` | Inference report |
| `chatbot/` | The engine shared by all three front ends |
| `data/` | Knowledge base, rules, intents, FAQs, test sets |
| `tools/` | Scripts that regenerate the report (`build_report.py`) and the notebook (`build_notebook.py`), plus quick test scripts |

## How to run (VS Code)
1. **File > Open Folder** and select this folder.
2. Notebook: open `University_Enquiry_Chatbot.ipynb`, pick the **Python 3.12** kernel (top right), then click **Run All**.
3. Terminal chatbot: **Terminal > New Terminal**, then run:
   ```
   python chat_cli.py
   ```
   Commands: `/why` (shows how the answer was produced), `/reset`, `/help`, `/exit`.
4. Web demo:
   ```
   streamlit run app_streamlit.py
   ```
   Then open http://localhost:8501. Use `?demo=1` or `?demo=2` to pre-load a sample conversation.

## Setup steps that were followed (Windows, all free, user account only)
```
winget install --id Python.Python.3.12 -e --scope user
winget install --id Microsoft.VisualStudioCode -e --scope user
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter
python -m pip install -r requirements.txt
```
The Python and `Python\Scripts` folders were added to the user PATH.

## Results (105-query test set)
| System | Answer accuracy | Intent macro-F1 |
|---|---|---|
| Keyword FAQ matching | 31.4% | 0.39 |
| NLP + vector retrieval | 38.1% | 0.49 |
| **Hybrid KR + NLP (this project)** | **92.4%** | **0.93** |

On the held-out set, which was scored once after development, the hybrid chatbot got 80.0% of answers right, against 48.0% and 32.0% for the two baselines.
