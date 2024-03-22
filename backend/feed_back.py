import csv
import os


def save_feedback_to_csv(
    message_id, response, feedback, csv_filename="feedback_data.csv"
):
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_filename)

    # Open the CSV file in append mode
    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["MessageID", "Response", "Feedback"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file doesn't exist, write the header row
        if not file_exists:
            writer.writeheader()

        # Write the data to the CSV file
        writer.writerow(
            {
                "MessageID": message_id,
                "Response": response,
                "Feedback": feedback,
            }
        )

    print(f"Feedback saved to {csv_filename}")


def save_report_to_csv(message_id, response, csv_filename="report_data.csv"):
    # Check if the CSV file exists
    file_exists = os.path.isfile(csv_filename)

    # Open the CSV file in append mode
    with open(csv_filename, "a", newline="") as csvfile:
        fieldnames = ["MessageID", "Response"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # If the file doesn't exist, write the header row
        if not file_exists:
            writer.writeheader()

        # Write the data to the CSV file
        writer.writerow(
            {
                "MessageID": message_id,
                "Response": response,
            }
        )

    print(f"Report saved to {csv_filename}")
