import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset

# Load the dataset
ds = load_dataset("laugustyniak/abusive-clauses-pl")

# Map labels to class names
label_map = {0: 'KLAUZULA_ABUZYWNA', 1: 'BEZPIECZNE_POSTANOWIENIE_UMOWNE'}

# Funkcja do wypisania rozkładu klas i ich nazw w zbiorach danych
def show_class_distribution(dataset, subsets):
    for subset in subsets:
        if subset in dataset:
            data = dataset[subset].to_pandas()
            data['label'] = data['label'].map(label_map)  # Zamiana 0/1 na nazwy klas
            print(f"Rozkład klas w zbiorze {subset}:")
            print(data['label'].value_counts(), "\n")  # Wypisanie liczby próbek dla każdej klasy

# Funkcja do tworzenia wykresu słupkowego dla rozkładu klas
def plot_class_distribution(dataset, subsets):
    for subset in subsets:
        if subset in dataset:
            data = dataset[subset].to_pandas()
            data['label'] = data['label'].map(label_map)  # Zamiana 0/1 na nazwy klas
            plt.figure(figsize=(12, 6))  # Większy wykres

            # Tworzenie dwóch wykresów w jednym
            plt.subplot(1, 2, 1)  # Pierwszy wykres (Rozkład klas)
            sns.countplot(data=data, x='label', hue='label', palette='Set2', alpha=0.8)
            plt.title(f"Rozkład klas w zbiorze {subset}")
            plt.xlabel("Klasy")
            plt.ylabel("Liczba próbek")
            plt.xticks(rotation=0, fontsize=10)

            # Obliczenia dla średniej liczby słów w każdej klasie
            data['word_count'] = data['text'].apply(lambda x: len(x.split()))
            avg_words_by_class = data.groupby('label')['word_count'].mean()

            # Drugi wykres (Średnia liczba słów w każdej klasie)
            plt.subplot(1, 2, 2)  # Drugi wykres (Średnia liczba słów w każdej klasie)
            sns.barplot(x=avg_words_by_class.index, y=avg_words_by_class.values, palette='Set2')
            plt.title(f"Średnia liczba słów w każdej klasie w zbiorze {subset}")
            plt.xlabel("Klasa")
            plt.ylabel("Średnia liczba słów")

            # Wyświetlenie wykresu
            plt.tight_layout()
            plt.show()

# Funkcja do obliczenia statystyk ogólnych i tworzenia wykresów
def show_word_statistics_and_plots(dataset, subsets):
    for subset in subsets:
        if subset in dataset:
            data = dataset[subset].to_pandas()
            data['label'] = data['label'].map(label_map)  # Zamiana 0/1 na nazwy klas

            # Policz liczbę słów w każdej próbce
            data['word_count'] = data['text'].apply(lambda x: len(x.split()))

            # Ogólne statystyki
            total_words = data['word_count'].sum()  # Całkowita liczba słów
            avg_words_per_sample = data['word_count'].mean()  # Średnia liczba słów na próbkę
            avg_words_by_class = data.groupby('label')['word_count'].mean()  # Średnia liczba słów w każdej klasie

            print(f"Statystyki dla zbioru {subset}:")
            print(f"Całkowita liczba słów: {total_words}")
            print(f"Średnia liczba słów na próbkę: {avg_words_per_sample:.2f}")
            print(f"Średnia liczba słów dla każdej klasy:")
            print(avg_words_by_class, "\n")

# Wywołanie funkcji
subsets = ["train", "validation", "test"]

# Wypisywanie rozkładu klas
show_class_distribution(ds, subsets)

# Tworzenie wykresów słupkowych
plot_class_distribution(ds, subsets)

# Wypisywanie statystyk ogólnych
show_word_statistics_and_plots(ds, subsets)
