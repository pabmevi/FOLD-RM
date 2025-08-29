from utils import load_data, split_data, get_scores, justify_data, decode_rules, proof_tree, scores, zip_rule, simplify_rule
from algo import foldrm, predict, classify, flatten_rules, justify, add_constraint
import pickle


class Classifier:
    def __init__(self, attrs=None, numeric=None, label=None):
        self.attrs = attrs
        self.numeric = numeric
        self.label = label
        self.rules = None
        self.frs = None
        self.crs = None
        self.asp_rules = None
        self.seq = 1
        self.simple = None
        self.translation = None

def confidence_fit(self, data, improvement_threshold=0.05, ratio=0.5, min_confidence=0.75):
        """
        Entrena el modelo y calcula la confianza de las reglas.
        - data: dataset
        - improvement_threshold: umbral para mejora (si usas expand_rules)
        - ratio: proporción de data para entrenamiento
        - min_confidence: confianza mínima para conservar la regla
        """
        # 1️⃣ Entrenar usando foldrm o expand_rules según corresponda
        if self.rules is None or len(self.rules) == 0:
            self.rules = foldrm(data, ratio=ratio)
        else:
            self.rules = expand_rules(data, existing_rules=self.rules, ratio=ratio, improvement_threshold=improvement_threshold)

        # 2️⃣ Calcular confianza para cada regla
        new_rules = []
        for rule in self.rules:
            # Supongamos que el número de aciertos (tp) y total están guardados en la regla
            tp = rule[-2] if len(rule) >= 2 else 0
            total = rule[-3] if len(rule) >= 3 else 1
            conf = calculate_confidence(tp, total)
            new_rules.append(rule + (conf,))

        # 3️⃣ Podar reglas con confianza menor que min_confidence
        self.rules = prune_rules(new_rules, confidence=min_confidence)

    def load_data(self, file, amount=-1):
        if self.label != self.attrs[-1]:
            data, self.attrs = load_data(file, self.attrs, self.label, self.numeric, amount)
        else:
            data, _ = load_data(file, self.attrs[:-1], self.label, self.numeric, amount)
        return data

    def fit(self, data, ratio=0.5):
        self.rules = foldrm(data, ratio=ratio)

    def predict(self, X):
        return predict(self.rules, X)

    def classify(self, x):
        return classify(self.rules, x)

    def asp(self, simple=False):
        if (self.asp_rules is None and self.rules is not None) or self.simple != simple:
            self.simple = simple
            self.frs = flatten_rules(self.rules)
            self.frs = [zip_rule(r) for r in self.frs]
            if simple:
                self.asp_rules = decode_rules(self.frs, self.attrs)
                self.asp_rules = [simplify_rule(r) for r in self.asp_rules]
            else:
                self.crs = add_constraint(self.frs)
                self.asp_rules = decode_rules(self.crs, self.attrs)
            self.crs = self.frs
        return self.asp_rules

    def print_asp(self, simple=False):
        for r in self.asp(simple):
            print(r)

    def explain(self, x):
        ret = ''
        pos = []
        justify(self.crs, x, pos=pos)
        expl = decode_rules(pos, self.attrs, x=x)
        for e in expl:
            ret = ret + e + '\n'
        ret = ret + str(justify_data(pos, x, attrs=self.attrs)) + '\n'
        return ret

    def proof(self, x):
        ret = ''
        pos = []
        justify(self.crs, x, pos=pos)
        expl = proof_tree(pos, self.attrs, x=x)
        for e in expl:
            ret = ret + e + '\n'
        ret = ret + str(justify_data(pos, x, attrs=self.attrs)) + '\n'
        return ret


def save_model_to_file(model, file):
    f = open(file, 'wb')
    pickle.dump(model, f)
    f.close()


def load_model_from_file(file):
    f = open(file, 'rb')
    ret = pickle.load(f)
    f.close()
    return ret

