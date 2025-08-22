# threat handling workflow
# each log in which prediction == 1 is flagged for review by chat
if __name__ == "__main__":
    for log, pred in zip(new_logs.to_dict(orient='records'), predictions):
        if pred == 1:
            # Send log to ChatGPT for advice (pseudo-code)
            response = chatgpt_api.ask(f"How should I respond to this threat? {log}")
            print(response)

