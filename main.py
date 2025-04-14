#!/usr/bin/env python3
import math

# ---------------------------
# Etap 1: Wczytywanie danych i podstawowe obliczenia
# ---------------------------
def load_data(file_name):
    """
    Wczytuje dane z pliku tekstowego, zakładając, że wartości są oddzielone przecinkami.
    Próbuje przekonwertować elementy do int, następnie do float, a w ostateczności zachowuje je jako string.
    Zwraca listę wierszy (każdy wiersz to lista wartości).
    """
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # pomijamy puste linie
                # Rozdzielenie wartości – zakładamy przecinek jako separator
                row_values = line.split(',')
                new_row = []
                for value in row_values:
                    value = value.strip()
                    try:
                        new_row.append(int(value))
                    except ValueError:
                        try:
                            new_row.append(float(value))
                        except ValueError:
                            new_row.append(value)
                data.append(new_row)
    return data

def calculate_attribute_stats(data):
    """
    Dla każdego atrybutu (kolumny) oblicza:
      - możliwe wartości
      - liczbę wystąpień poszczególnych wartości.
    Zwraca słownik: {indeks_kolumny: {wartość: liczba_wystąpień}}.
    """
    stats = {}
    num_attributes = len(data[0])
    for i in range(num_attributes):
        stats[i] = {}
    for row in data:
        for i, value in enumerate(row):
            stats[i][value] = stats[i].get(value, 0) + 1
    return stats

def calculate_entropy(counts):
    """
    Oblicza entropię według wzoru:
        I(P) = - sum(p * log2(p))
    gdzie counts to słownik zawierający liczbę wystąpień poszczególnych wartości.
    """
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy

def calculate_attribute_info(data, attribute_index):
    """
    Oblicza funkcję informacji dla atrybutu o zadanym indeksie według wzoru:
        Info(X, T) = sum(|T_i|/|T| * Info(T_i))
    gdzie T_i to zbiór przypadków, które mają daną wartość atrybutu X,
    a Info(T_i) to entropia rozkładu klas decyzyjnych (ostatnia kolumna) w T_i.
    """
    total = len(data)
    groups = {}
    for row in data:
        value = row[attribute_index]
        groups.setdefault(value, []).append(row)

    info = 0.0
    for subset in groups.values():
        decision_counts = {}
        for r in subset:
            dec_value = r[-1]  # ostatnia kolumna jako atrybut decyzyjny
            decision_counts[dec_value] = decision_counts.get(dec_value, 0) + 1
        subset_entropy = calculate_entropy(decision_counts)
        info += (len(subset) / total) * subset_entropy
    return info

def calculate_gain(data, attribute_index, decision_entropy):
    """
    Oblicza przyrost informacji dla atrybutu według wzoru:
        Gain(X, T) = Info(T) - Info(X, T)
    """
    attribute_info = calculate_attribute_info(data, attribute_index)
    return decision_entropy - attribute_info

def calculate_split_info(data, attribute_index):
    """
    Oblicza SplitInfo dla atrybutu według wzoru:
        SplitInfo(X, T) = I( |T1|/|T|, |T2|/|T|, ..., |Tk|/|T| )
    czyli entropię rozkładu częstości występowania wartości danego atrybutu.
    """
    counts = {}
    for row in data:
        value = row[attribute_index]
        counts[value] = counts.get(value, 0) + 1
    return calculate_entropy(counts)

def calculate_gain_ratio(data, attribute_index, decision_entropy):
    """
    Oblicza zrównoważony przyrost informacji (Gain Ratio) według wzoru:
        GainRatio(X, T) = Gain(X, T) / SplitInfo(X, T)
    Jeśli SplitInfo wynosi 0, zwraca 0.
    """
    gain = calculate_gain(data, attribute_index, decision_entropy)
    split_info = calculate_split_info(data, attribute_index)
    if split_info == 0:
        return 0
    return gain / split_info

# ---------------------------
# Etap 2: Rekurencyjna budowa drzewa decyzyjnego (rozwiązanie rozdziału 7)
# ---------------------------
class TreeNode:
    def __init__(self, attribute=None, children=None, label=None):
        """
        attribute: indeks atrybutu używanego do podziału w tym węźle (dla węzła wewnętrznego)
        children: słownik {wartość_atrybutu: TreeNode} – kolejne poddrzewo
        label: etykieta klasy (dla węzła liściastego)
        """
        self.attribute = attribute
        self.children = children if children is not None else {}
        self.label = label

def build_tree(data):
    """
    Rekurencyjnie buduje drzewo decyzyjne na podstawie danych:
      - Jeśli wszystkie przypadki mają tę samą klasę, zwraca węzeł liściasty.
      - Dla pozostałych przypadków wybiera atrybut o najwyższym Gain Ratio.
      - Jeśli najlepszy Gain Ratio wynosi 0, zwraca węzeł liściasty z etykietą dominującą.
      - Dla każdej wartości wybranego atrybutu tworzy oddzielny podzbiór i rekurencyjnie buduje dla niego poddrzewo.
    """
    if not data:
        return None

    # Sprawdzamy, czy wszystkie przypadki mają tę samą decyzję
    decisions = [row[-1] for row in data]
    if len(set(decisions)) == 1:
        return TreeNode(label=decisions[0])

    # Obliczamy rozkład klas decyzyjnych i entropię zbioru
    decision_counts = {}
    for dec in decisions:
        decision_counts[dec] = decision_counts.get(dec, 0) + 1
    decision_entropy = calculate_entropy(decision_counts)

    num_attributes = len(data[0]) - 1  # ostatnia kolumna to decyzja
    best_attribute = None
    best_gain_ratio = -1.0

    # Wybieramy najlepszy atrybut według Gain Ratio
    for i in range(num_attributes):
        gr = calculate_gain_ratio(data, i, decision_entropy)
        if gr > best_gain_ratio:
            best_gain_ratio = gr
            best_attribute = i

    # Kryterium zakończenia: brak przyrostu informacji – tworzymy liść
    if best_gain_ratio == 0:
        majority_label = max(decision_counts, key=decision_counts.get)
        return TreeNode(label=majority_label)

    # Tworzymy węzeł wewnętrzny – test dla wybranego atrybutu
    node = TreeNode(attribute=best_attribute)
    values = set(row[best_attribute] for row in data)
    for value in values:
        subset = [row for row in data if row[best_attribute] == value]
        node.children[value] = build_tree(subset)
    return node

def print_tree(node, indent_level=0, value_prefix=""):
    """
    Rekurencyjnie wypisuje strukturę drzewa decyzyjnego w formacie zgodnym z wizualizacja.txt.
    """
    indent = "          " * indent_level # 10 spacji na poziom

    if node is None:
        print(indent + value_prefix + "Brak drzewa")
        return

    if node.label is not None:
        # Format liścia: <wcięcie><prefix_wartości> -> D: <etykieta>
        print(indent + value_prefix + " -> D: " + str(node.label))
    else:
        # Format węzła wewnętrznego
        attribute_display_index = node.attribute + 1 # Dostosuj indeks do wyświetlania (od 1)
        if indent_level == 0:
            # Format korzenia: Atrybut: <indeks>
            print("Atrybut: " + str(attribute_display_index))
        else:
            # Format węzła wewnętrznego: <wcięcie><prefix_wartości>->Atrybut: <indeks>
            print(indent + value_prefix + "->Atrybut: " + str(attribute_display_index))

        # Sortuj dzieci według wartości dla spójnego wyniku
        # Konwertuj klucze na stringi do sortowania, aby obsłużyć potencjalne mieszane typy (int/str)
        sorted_children = sorted(node.children.items(), key=lambda item: str(item[0]))

        for value, child in sorted_children:
            # Rekurencyjne wywołanie dla dzieci
            # prefix_wartości dla następnego poziomu to wartość bieżącej gałęzi
            print_tree(child, indent_level + 1, value_prefix=str(value))

def main():    
    file_name = "Dane testowe/gielda.txt"

    print(f"Wczytywanie danych z pliku: {file_name}")
    data = load_data(file_name)
    
    print("Wczytane dane:")
    for row in data:
        print(row)

    stats = calculate_attribute_stats(data)
    num_attributes = len(stats)

    print("\nStatystyki atrybutów:")
    for i, stat_data in stats.items():        
        display_index = i + 1
        if i == num_attributes - 1:
            print(f"Atrybut decyzyjny (kolumna {display_index}):")
        else:
            print(f"Atrybut {display_index}:")
        for value, count in stat_data.items():
            print(f"  Wartość {value}: {count}")

    # Obliczenie entropii zbioru (na podstawie atrybutu decyzyjnego)
    decision_attribute_stats = stats[num_attributes - 1]
    decision_entropy = calculate_entropy(decision_attribute_stats)
    print("\nEntropia zbioru (atrybut decyzyjny): {:.4f}".format(decision_entropy))

    # Obliczenia Info, Gain, SplitInfo, Gain Ratio dla atrybutów warunkowych
    print("\nObliczenia dla atrybutów warunkowych:")
    for i in range(num_attributes - 1):        
        display_index = i + 1
        info_attr = calculate_attribute_info(data, i)
        gain = calculate_gain(data, i, decision_entropy)
        split_info = calculate_split_info(data, i)
        gain_ratio = calculate_gain_ratio(data, i, decision_entropy)
        print(f"\nAtrybut {display_index}:")
        print(f"  Info(a{display_index}, T)     = {info_attr}")
        print(f"  Gain(a{display_index}, T)     = {gain}")
        print(f"  SplitInfo(a{display_index}, T)= {split_info}")
        print(f"  GainRatio(a{display_index}, T)= {gain_ratio}")

    # Budowa drzewa decyzyjnego
    print(f"\nDrzewo dla pliku {file_name}")
    tree = build_tree(data)
    print_tree(tree)

if __name__ == "__main__":
    main()
