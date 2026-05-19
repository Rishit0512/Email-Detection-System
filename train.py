from __future__ import annotations

import os
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")


def load_embedded_dataset() -> Tuple[List[str], List[int]]:
    
    spam_texts = [
        "URGENT! Your account has been suspended. Verify your identity immediately to avoid penalties.",
        "Account locked due to suspicious activity. Please confirm your password immediately.",
        "Security alert: Suspicious login detected. Click to verify your account now.",
        "You have been selected for a limited offer. Act now and get guaranteed results.",
        "Congratulations! You are the winner of a FREE prize. Claim your reward today.",
        "Winner! Get a free gift card. Provide your details to receive the prize.",
        "Claim your reward—no cost. We noticed unusual activity on your account. Verify today.",
        "Your account needs verification. Sign in securely to prevent account freeze.",
        "Final notice: Your subscription will be blocked unless you pay the outstanding invoice.",
        "Payment overdue. Please settle immediately to avoid service interruption.",
        "Invoice attached. Respond now to prevent suspension.",
        "Dear customer, wire transfer required. Contact support for immediate assistance.",
        "Bitcoin investment opportunity with guaranteed returns. Limited time only!",
        "Limited time! Invest now and earn guaranteed profits.",
        "Act fast: your reward is waiting. Verify your account to receive it.",
        "We noticed unusual transactions. Login to your account and confirm details.",
        "Your wallet has been flagged. Transfer required to secure funds.",
        "Congratulations. You have won a prize. Click the link to confirm.",
        "Important: Your account will be terminated unless you verify within 24 hours.",
        "This is a final warning. Update your payment information immediately.",
    ]

    safe_texts = [
        "Hi team, the meeting is scheduled for tomorrow at 10 AM. Please review the agenda.",
        "Your order has shipped. Tracking information will be sent once available.",
        "Reminder: submit your timesheet by Friday. Let me know if you have questions.",
        "Weekly newsletter: new updates and features now live on the dashboard.",
        "Invoice attached for your records. Payment due on the usual schedule.",
        "Password change confirmation: Your password was updated successfully.",
        "Following up on our call. I will send the documents you requested.",
        "Customer support reply: We are happy to help with your request.",
        "Project status update: tasks are on track for the milestone.",
        "Invitation: You are welcome to join the webinar next week.",
        "Receipt confirmed. Thank you for your purchase.",
        "FYI: The report draft is ready. Please review when you can.",
        "Can you confirm the updated delivery date for my order?",
        "Thanks for your message. I will get back to you shortly.",
        "Your subscription is renewed successfully. No action is required.",
        "We received your payment. Your account remains active.",
        "Team update: Deployment completed and all services are running.",
        "Agenda for today's standup is attached. See you at 9:30.",
        "Here are the meeting notes from yesterday: action items listed below.",
    ]


    texts = spam_texts + safe_texts
    labels = [1] * len(spam_texts) + [0] * len(safe_texts)
    return texts, labels


def build_pipeline() -> Pipeline:

    clf = LogisticRegression(max_iter=4000, class_weight="balanced", C=1.0)

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.9,
                ),
            ),
            ("clf", clf),
        ]
    )



def main():
    texts, labels = load_embedded_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    acc = pipeline.score(X_test, y_test)
    
    vectorizer = pipeline.named_steps["tfidf"]
    feature_names = list(vectorizer.get_feature_names_out())

    joblib.dump(
        {"pipeline": pipeline, "feature_names": feature_names, "acc": float(acc)},
        MODEL_PATH,
    )

    print(f"Trained model saved to: {MODEL_PATH} (quick holdout acc={acc:.3f})")


if __name__ == "__main__":
    main()

