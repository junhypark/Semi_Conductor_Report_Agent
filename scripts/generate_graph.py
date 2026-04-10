from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.supervisor.graph import save_graph_mermaid_source, save_graph_visualization


def main() -> None:
    xray_path = save_graph_visualization()
    xray_mermaid_path = save_graph_mermaid_source()
    print(xray_path)
    print(xray_mermaid_path)


if __name__ == "__main__":
    main()
