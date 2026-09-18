from collections import defaultdict
import gradio as gr
from dotenv import load_dotenv
import pandas as pd

from implementation.answer import answer_question
from evaluation.eval import evaluate_all_retrieval, evaluate_all_answers

load_dotenv(override=True)

MRR_GREEN = 0.9
MRR_AMBER = 0.75
NDCG_GREEN = 0.9
NDCG_AMBER = 0.75
COVERAGE_GREEN = 90.0
COVERAGE_AMBER = 75.0

ANSWER_GREEN = 4.5
ANSWER_AMBER = 4.0


def get_color(value: float, metric_type: str) -> str:
  """Get color based on metric value and type."""
  if metric_type == "mrr":
    if value >= MRR_GREEN:
      return "green"
    elif value >= MRR_AMBER:
      return "orange"
    else:
      return "red"
  elif metric_type == "ndcg":
    if value >= NDCG_GREEN:
      return "green"
    elif value >= NDCG_AMBER:
      return "orange"
    else:
      return "red"
  elif metric_type == "coverage":
    if value >= COVERAGE_GREEN:
      return "green"
    elif value >= COVERAGE_AMBER:
      return "orange"
    else:
      return "red"
  elif metric_type in ["accuracy", "completeness", "relevance"]:
    if value >= ANSWER_GREEN:
      return "green"
    elif value >= ANSWER_AMBER:
      return "orange"
    else:
      return "red"
  return "black"


def format_metric_html(
    label: str,
    value: float,
    metric_type: str,
    is_percentage: bool = False,
    score_format: bool = False,
) -> str:
  """Format a metric with color coding."""
  color = get_color(value, metric_type)
  if is_percentage:
    value_str = f"{value:.1f}%"
  elif score_format:
    value_str = f"{value:.2f}/5"
  else:
    value_str = f"{value:.4f}"
  return f"""
    <div style="margin: 10px 0; padding: 15px; background-color: #f5f5f5; border-radius: 8px; border-left: 5px solid {color};">
        <div style="font-size: 14px; color: #666; margin-bottom: 5px;">{label}</div>
        <div style="font-size: 28px; font-weight: bold; color: {color};">{value_str}</div>
    </div>
    """


def chat_response(message, history):
  """Handle chat interactions with the RAG pipeline using a standard chatbot list format."""
  if not message.strip():
    return history, ""
  
  # Append user message to history
  history = history or []
  history.append({"role": "user", "content": message})
  
  try:
    # Get answer and chunks from backend
    answer, chunks = answer_question(message, history)
    history.append({"role": "assistant", "content": answer})
  except Exception as e:
    history.append({"role": "assistant", "content": f"Error generating response: {str.format(str(e))}"})
    
  return history, ""


def run_retrieval_evaluation(progress=gr.Progress()):
  """Run retrieval evaluation and yield updates."""
  total_mrr = 0.0
  total_ndcg = 0.0
  total_coverage = 0.0
  category_mrr = defaultdict(list)
  count = 0

  for test, result, prog_value in evaluate_all_retrieval():
    count += 1
    total_mrr += result.mrr
    total_ndcg += result.ndcg
    total_coverage += result.keyword_coverage

    category_mrr[test.category].append(result.mrr)

    progress(prog_value, desc=f"Evaluating wrestling test {count}...")

  avg_mrr = total_mrr / count
  avg_ndcg = total_ndcg / count
  avg_coverage = total_coverage / count

  final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Mean Reciprocal Rank (MRR)", avg_mrr, "mrr")}
        {format_metric_html("Normalized DCG (nDCG)", avg_ndcg, "ndcg")}
        {format_metric_html("Wrestler/Show Keyword Coverage", avg_coverage, "coverage", is_percentage=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

  category_data = []
  for category, mrr_scores in category_mrr.items():
    avg_cat_mrr = sum(mrr_scores) / len(mrr_scores)
    category_data.append({"Category": category, "Average MRR": avg_cat_mrr})

  df = pd.DataFrame(category_data)

  return final_html, df


def run_answer_evaluation(progress=gr.Progress()):
  """Run answer evaluation and yield updates (async)."""
  total_accuracy = 0.0
  total_completeness = 0.0
  total_relevance = 0.0
  category_accuracy = defaultdict(list)
  count = 0

  for test, result, prog_value in evaluate_all_answers():
    count += 1
    total_accuracy += result.accuracy
    total_completeness += result.completeness
    total_relevance += result.relevance

    category_accuracy[test.category].append(result.accuracy)

    progress(prog_value, desc=f"Evaluating wrestling test {count}...")

  avg_accuracy = total_accuracy / count
  avg_completeness = total_completeness / count
  avg_relevance = total_relevance / count

  final_html = f"""
    <div style="padding: 0;">
        {format_metric_html("Match Result Accuracy", avg_accuracy, "accuracy", score_format=True)}
        {format_metric_html("Summary Completeness", avg_completeness, "completeness", score_format=True)}
        {format_metric_html("Promo/Show Relevance", avg_relevance, "relevance", score_format=True)}
        <div style="margin-top: 20px; padding: 10px; background-color: #d4edda; border-radius: 5px; text-align: center; border: 1px solid #c3e6cb;">
            <span style="font-size: 14px; color: #155724; font-weight: bold;">✓ Evaluation Complete: {count} tests</span>
        </div>
    </div>
    """

  category_data = []
  for category, accuracy_scores in category_accuracy.items():
    avg_cat_accuracy = sum(accuracy_scores) / len(accuracy_scores)
    category_data.append({"Category": category, "Average Accuracy": avg_cat_accuracy})

  df = pd.DataFrame(category_data)

  return final_html, df


def main():
  """Launch the Gradio app with fixed compatibility."""
  theme = gr.themes.Soft(font=["Inter", "system-ui", "sans-serif"])

  with gr.Blocks(title="WWEish AI Assistant & Dashboard") as app:
    gr.Markdown("# 🤼‍♂️ WWEish AI Assistant")
    gr.Markdown("Chat with your AI assistant about WWE matches, history, and events.")

    chatbot = gr.Chatbot(height=400)
    with gr.Row():
      msg_input = gr.Textbox(
          placeholder="Ask anything about WWE 2023 matches...",
          container=False,
          scale=8,
      )
      submit_btn = gr.Button("Send", variant="primary", scale=1)

    msg_input.submit(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])
    submit_btn.click(chat_response, inputs=[msg_input, chatbot], outputs=[chatbot, msg_input])

    gr.Markdown("---")
    gr.Markdown("# 📊 WWEish AI Evaluation Dashboard")
    gr.Markdown("Evaluate retrieval context and match summary/promo quality for the WWEish AI Assistant")

    with gr.Row():
      with gr.Column(scale=1):
        gr.Markdown("## 🔍 Context Retrieval")
        retrieval_button = gr.Button("Run Retrieval Eval", variant="primary")
        retrieval_metrics = gr.HTML("<div style='padding: 10px; text-align: center; color: #999;'>Click to start</div>")
        retrieval_chart = gr.BarPlot(
            x="Category",
            y="Average MRR",
            title="Avg MRR by Category",
            y_lim=[0, 1],
            height=300,
        )

      with gr.Column(scale=1):
        gr.Markdown("## 💬 Answer Evaluation")
        answer_button = gr.Button("Run Answer Eval", variant="primary")
        answer_metrics = gr.HTML("<div style='padding: 10px; text-align: center; color: #999;'>Click to start</div>")
        answer_chart = gr.BarPlot(
            x="Category",
            y="Average Accuracy",
            title="Avg Accuracy by Category",
            y_lim=[1, 5],
            height=300,
        )

    retrieval_button.click(
        fn=run_retrieval_evaluation,
        outputs=[retrieval_metrics, retrieval_chart],
    )

    answer_button.click(
        fn=run_answer_evaluation,
        outputs=[answer_metrics, answer_chart],
    )

  app.launch(inbrowser=True, theme=theme)
if __name__ == "__main__":
  main()



# import gradio as gr
# from dotenv import load_dotenv
# from implementation.answer import answer_question

# load_dotenv(override=True)

# def chat_response(message, history):
#     """Handle chat interactions simply."""
#     if not message.strip():
#         return history
    
#     history = history or []
#     history.append({"role": "user", "content": message})
    
#     try:
#         answer, chunks = answer_question(message, history)
#         history.append({"role": "assistant", "content": answer})
#     except Exception as e:
#         history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        
#     return history

# def main():
#     with gr.Blocks(title="WWEish AI Assistant") as app:
#         gr.Markdown("# 🤼‍♂️ WWEish AI Assistant")
        
#         chatbot = gr.Chatbot(height=450)
#         msg = gr.Textbox(placeholder="Ask anything about WWE...", container=False)
        
#         msg.submit(chat_response, inputs=[msg, chatbot], outputs=[chatbot])
        
#     app.launch(inbrowser=True)

# if __name__ == "__main__":
#     main()