from data.loader import load_pages
from graph.builder import build_graphs
from metrics.information_retention import evaluate_information_retention
pages = list(load_pages("doclaynet", limit=5))
graphs = build_graphs(pages)
print("Evaluating...")
acc = evaluate_information_retention(graphs)
print(f"Accuracy: {acc}")
