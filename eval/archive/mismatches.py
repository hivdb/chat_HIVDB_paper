import pandas as pd

DETAIL_PATH = "eval/detailed_evaluation.csv"
OUTPUT_PATH = "eval/archive/mismatch_report_gpt5.txt"

def main() -> None:
    detail = pd.read_csv(DETAIL_PATH)
    human = detail[detail["Scenario"] == "Human Answer"].copy()

    mismatches = human[human["GPT-5 base Correct"] == 0]
    columns = [
        "PMID",
        "QID",
        "Question",
        "Human Answer",
        "GPT-5 base Answer",
        "GPT-5 base Correct",
    ]
    output = mismatches[columns].to_string(index=False)
    print(output)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as outfile:
        outfile.write(output + "\n")


if __name__ == "__main__":
    main()
