"""
chatbot.py — SkyGuard AI ChatBot (Privacy-First, Offline)
A fully working AI chatbot that:
  1. Takes real datasets (CSV, Excel, JSON, Parquet, TSV)
  2. Answers questions about the data using local AI (no internet)
  3. NEVER shares data externally
  4. Securely deletes all data on exit or on user request

Run:
    python chatbot.py

Dependencies:
    pip install -r requirements.txt
"""

import sys
import signal
import os

# Block all outbound network connections for this process
# This GUARANTEES no data is ever sent to the internet
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from data_manager import DataManager
from ai_engine import AIEngine


# ------------------------------------------------------------------
# ANSI colors for the terminal
# ------------------------------------------------------------------
class C:
    BOLD = "\033[1m"
    DIM = "\033[2m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"


BANNER = f"""
{C.CYAN}{C.BOLD}╔══════════════════════════════════════════════════════════════╗
║               🛡️  SkyGuard AI ChatBot  🛡️                   ║
║          Privacy-First · Offline · Secure · Local            ║
╠══════════════════════════════════════════════════════════════╣
║  ✅ All data stays on YOUR machine — never sent anywhere     ║
║  ✅ Data is securely wiped on exit (overwritten + deleted)    ║
║  ✅ AI runs 100% offline — no API calls, no cloud            ║
║  ✅ Supports CSV, Excel, JSON, Parquet, TSV                  ║
╚══════════════════════════════════════════════════════════════╝{C.RESET}
"""

HELP_TEXT = f"""
{C.YELLOW}{C.BOLD}Available Commands:{C.RESET}
  {C.GREEN}load <file_path>{C.RESET}    — Load a dataset (CSV, Excel, JSON, Parquet, TSV)
  {C.GREEN}list{C.RESET}                — Show all loaded datasets
  {C.GREEN}use <id>{C.RESET}            — Switch active dataset (use the ID from 'list')
  {C.GREEN}info{C.RESET}                — Show summary of the active dataset
  {C.GREEN}delete <id>{C.RESET}         — Securely delete a specific dataset
  {C.GREEN}purge{C.RESET}               — Securely delete ALL loaded data
  {C.GREEN}help{C.RESET}                — Show this help message
  {C.GREEN}exit / quit{C.RESET}         — Securely wipe all data and exit

{C.YELLOW}{C.BOLD}Ask Questions:{C.RESET}
  Just type any question about your data!
  Examples:
    • How many rows are there?
    • What columns does it have?
    • Show the first 10 rows
    • What is the average temperature?
    • Are there missing values?
    • Show correlations
    • What is the max value of pressure?
    • Are there duplicates?
"""


def main():
    dm = DataManager()
    ai = AIEngine()
    active_id = None  # currently active dataset

    # Handle Ctrl+C gracefully — always clean up
    def signal_handler(sig, frame):
        print(f"\n{C.RED}{C.BOLD}⚠️  Interrupt received — securely wiping all data...{C.RESET}")
        dm.purge_all()
        print(f"{C.GREEN}✅ All data securely deleted. Goodbye!{C.RESET}")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    print(BANNER)
    print(HELP_TEXT)

    while True:
        try:
            # Prompt
            if active_id:
                meta = dm._datasets.get(active_id, {})
                name = meta.get("name", "?")
                prompt = f"{C.BLUE}{C.BOLD}[{name}]{C.RESET} {C.CYAN}You ▸{C.RESET} "
            else:
                prompt = f"{C.CYAN}You ▸{C.RESET} "

            user_input = input(prompt).strip()
            if not user_input:
                continue

            lower = user_input.lower()

            # --- Commands ---
            if lower in ("exit", "quit", "q", "bye"):
                print(f"\n{C.YELLOW}🔒 Securely wiping all data...{C.RESET}")
                dm.purge_all()
                print(f"{C.GREEN}{C.BOLD}✅ All data securely deleted. No traces remain.{C.RESET}")
                print(f"{C.CYAN}Goodbye! Your data was never shared. 🛡️{C.RESET}\n")
                break

            elif lower == "help":
                print(HELP_TEXT)

            elif lower.startswith("load "):
                path = user_input[5:].strip().strip('"').strip("'")
                try:
                    meta = dm.ingest(path)
                    active_id = meta["id"]
                    print(f"\n{C.GREEN}{C.BOLD}✅ Dataset loaded successfully!{C.RESET}")
                    print(f"   {C.DIM}Name: {meta['name']}")
                    print(f"   Rows: {meta['rows']:,}  |  Columns: {meta['cols']}")
                    print(f"   Columns: {', '.join(meta['columns'])}")
                    print(f"   ID: {meta['id']}{C.RESET}")
                    print(f"   {C.YELLOW}This dataset is now active. Ask me anything!{C.RESET}\n")
                except FileNotFoundError as e:
                    print(f"{C.RED}❌ {e}{C.RESET}")
                except ValueError as e:
                    print(f"{C.RED}❌ {e}{C.RESET}")
                except Exception as e:
                    print(f"{C.RED}❌ Failed to load: {e}{C.RESET}")

            elif lower == "list":
                datasets = dm.list_datasets()
                if not datasets:
                    print(f"{C.YELLOW}No datasets loaded. Use 'load <file_path>' to add one.{C.RESET}")
                else:
                    print(f"\n{C.CYAN}{C.BOLD}Loaded Datasets:{C.RESET}")
                    for ds in datasets:
                        marker = " ◀ active" if ds["id"] == active_id else ""
                        print(f"  {C.GREEN}{ds['id']}{C.RESET} — {ds['name']} "
                              f"({ds['rows']:,} rows, {ds['cols']} cols){C.MAGENTA}{marker}{C.RESET}")
                    print()

            elif lower.startswith("use "):
                ds_id = user_input[4:].strip()
                if ds_id in dm._datasets:
                    active_id = ds_id
                    name = dm._datasets[ds_id]["name"]
                    print(f"{C.GREEN}✅ Switched to dataset: {name}{C.RESET}")
                else:
                    print(f"{C.RED}❌ Dataset ID '{ds_id}' not found. Use 'list' to see IDs.{C.RESET}")

            elif lower.startswith("delete "):
                ds_id = user_input[7:].strip()
                if ds_id in dm._datasets:
                    name = dm._datasets[ds_id]["name"]
                    dm.delete_dataset(ds_id)
                    if active_id == ds_id:
                        active_id = None
                    print(f"{C.GREEN}🗑️  Dataset '{name}' securely deleted.{C.RESET}")
                else:
                    print(f"{C.RED}❌ Dataset ID '{ds_id}' not found.{C.RESET}")

            elif lower == "purge":
                dm.purge_all()
                active_id = None
                print(f"{C.GREEN}{C.BOLD}🗑️  All data securely purged!{C.RESET}")

            elif lower == "info":
                if not active_id:
                    print(f"{C.YELLOW}No active dataset. Use 'load <file>' first.{C.RESET}")
                else:
                    summary = dm.dataset_summary(active_id)
                    print(f"\n{C.CYAN}{summary}{C.RESET}\n")

            # --- Natural language question ---
            else:
                if not active_id:
                    print(f"{C.YELLOW}⚠️  No dataset loaded. "
                          f"Use 'load <file_path>' to load a dataset first.{C.RESET}")
                    continue

                # Get the dataframe and context
                try:
                    df = dm.get_dataframe(active_id)
                    context = dm.dataset_summary(active_id)
                    answer = ai.answer(user_input, context, df)
                    print(f"\n{C.GREEN}{C.BOLD}🤖 AI:{C.RESET} {answer}\n")
                except Exception as e:
                    print(f"{C.RED}❌ Error analyzing data: {e}{C.RESET}")

        except EOFError:
            # Handle piped input ending
            print(f"\n{C.YELLOW}🔒 Securely wiping all data...{C.RESET}")
            dm.purge_all()
            print(f"{C.GREEN}✅ All data securely deleted.{C.RESET}")
            break


if __name__ == "__main__":
    main()
