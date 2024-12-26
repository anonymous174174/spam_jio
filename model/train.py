import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pickle


def train_model(data_file: str, output_model_file: str):
    """
    Train a text classification model and save it to disk.
    """
    # Load the dataset
    dataset = pd.read_csv(data_file)
    features, labels = dataset["text"], dataset["label"]

    # Split the dataset into training and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )

    # Define the vectorizer and classifier
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()

    # Convert text to feature vectors
    train_vectors = vectorizer.fit_transform(train_features)
    test_vectors = vectorizer.transform(test_features)

    # Train the classifier
    classifier.fit(train_vectors, train_labels)

    # Evaluate the classifier
    predictions = classifier.predict(test_vectors)
    accuracy = accuracy_score(test_labels, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")

    # Save the trained components
    with open(output_model_file, "wb") as model_file:
        pickle.dump((vectorizer, classifier), model_file)
    print(f"Model saved to {output_model_file}")


if __name__ == "__main__":
    train_model("dataset.csv", "model/naive_bayes_model.pkl")
