import pandas as pd
from modules.dataset_loader import DatasetLoader


def get_and_print_distribution(loader: DatasetLoader, task_ids):
    all_records = []

    print(f"Starting to process {len(task_ids)} tasks...")

    for tid in task_ids:
        # Get task data.
        data = loader.get_task(task_id=tid, include_goal=True, lazy=True)

        # Safety check: ensure goal_details exists.
        if 'goal_details' in data:
            details = data['goal_details']

            # 1. Extract the meta dictionary. Use copy to avoid modifying source data.
            # If meta does not exist, use an empty dictionary.
            record = details.get('meta', {}).copy()

            # 2. Extract goal_type and merge it into record.
            # Based on the screenshot, goal_type is directly under goal_details.
            record['goal_type'] = details['goal_details'].get('goal_type', 'Unknown')

            all_records.append(record)

    # Convert to DataFrame.
    df = pd.DataFrame(all_records)

    # 3. Count and print distributions.
    print("\n" + "=" * 30)
    print("      Field Distribution Statistics      ")
    print("=" * 30 + "\n")

    if df.empty:
        print("No data extracted, please check if task_ids are correct.")
        return

    # Iterate through each column for statistics.
    for column in df.columns:
        print(f"📊 Field: [{column}]")

        # Get the first non-empty value in this column to infer its type.
        first_valid_value = df[column].dropna().iloc[0] if not df[column].dropna().empty else None

        # Check whether the value is a list, such as plan_level or coor_level values like ['L0', 'L1'].
        if isinstance(first_valid_value, list):
            # explode() expands lists, such as one row ['A', 'B'] into two rows: 'A' and 'B'.
            # This counts total label occurrences instead of combination occurrences.
            distribution = df[column].explode().value_counts(normalize=False)
            total_count = distribution.sum()
            print(f"(List type detected, expanded for statistics)")
        else:
            # Count normal numeric or string values directly.
            distribution = df[column].value_counts(normalize=False)
            total_count = distribution.sum()

        # Print detailed counts and percentages.
        print("-" * 20)
        print(f"{'Value':<20} | {'Count':<10} | {'Percentage'}")
        print("-" * 50)

        for value, count in distribution.items():
            percent = (count / total_count) * 100
            print(f"{str(value):<20} | {count:<10} | {percent:.2f}%")

        print("\n")


# Run the main function.
if __name__ == "__main__":
    task_ids = ['cybertown_scenario_1_g_67011']  # Example; replace with the full list for actual use.
    dataset_loader = DatasetLoader(repo_id='WindyLab/GSI')
    get_and_print_distribution(loader=dataset_loader, task_ids=task_ids)
